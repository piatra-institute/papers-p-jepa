"""Minimal cellular sheaf over a learned cover of dishworld contexts.

This is the H4 hypothesis test from docs/DELIVERY_PLAN.md and the plan
file at ~/.claude/plans/ok-make-a-plan-declarative-scone.md. It is the
first place in the project where a real cellular sheaf is constructed:
a learned cover with K vertices, a 1-skeleton built from non-empty
overlaps, learned linear restriction maps per edge, an assembled
coboundary operator, a sheaf Laplacian, and reported dim H^0 / dim H^1.

The construction:

  - Vertices: K cluster centers c_i in R^D (D = len(action_features)).
  - Edges (i, j): non-empty if there is at least one stream context that
    lies within `overlap_radius` of both c_i and c_j (Euclidean).
  - Edge stalks: R^D (same dimension as vertex stalks).
  - Restriction maps rho_{i,ij} in R^{D x D} learned per edge by ridge
    least squares so that rho_{i,ij}(c_i) ~ m_{ij} ~ rho_{j,ij}(c_j),
    where m_{ij} is the empirical mean of overlap features.
  - Coboundary delta_0 assembled as a block matrix of shape (E*D, K*D)
    with rho_i on the i-th vertex block and -rho_j on the j-th.
  - Sheaf Laplacian L_0 = delta_0^T delta_0 acting on bigoplus F(i).
  - dim H^0 = K*D - rank(delta_0); dim H^1 = E*D - rank(delta_0) on the
    1-skeleton.
  - Glued sections: argmin_sigma sum_{ij} ||rho_i sigma_i - m_{ij}||^2
                                + sum_{ij} ||rho_j sigma_j - m_{ij}||^2
                                + gamma ||sigma - c||^2
    solved as one linear system.

The scalar baseline is the same K centers used unmodified. The
comparison is whether the sheaf-glued centers produce better downstream
action choice on held-out contexts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Cover:
    centers: np.ndarray  # (K, D)
    train_assignments: np.ndarray  # (N_train,) int in [0, K)

    @property
    def K(self) -> int:
        return int(self.centers.shape[0])

    @property
    def D(self) -> int:
        return int(self.centers.shape[1])


@dataclass(frozen=True)
class SheafEdges:
    edges: tuple[tuple[int, int], ...]
    overlap_counts: dict[tuple[int, int], int]
    overlap_means: dict[tuple[int, int], np.ndarray]

    @property
    def E(self) -> int:
        return len(self.edges)


@dataclass(frozen=True)
class SheafFit:
    restrictions: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]
    delta_0: np.ndarray
    sheaf_laplacian: np.ndarray
    rank_delta: int
    dim_h0: int
    dim_h1: int
    coboundary_energy_initial: float
    coboundary_energy_after_glue: float
    glued_centers: np.ndarray


def kmeans_cover(
    features: np.ndarray,
    k: int,
    seed: int,
    max_iter: int = 80,
    restarts: int = 12,
) -> Cover:
    """Deterministic k-means returning a Cover."""
    rng = np.random.default_rng(seed)
    best_inertia = np.inf
    best_centers: np.ndarray | None = None
    best_assignments: np.ndarray | None = None
    n = features.shape[0]
    for restart in range(restarts):
        init_idx = rng.choice(n, size=k, replace=False)
        centers = features[init_idx].copy()
        prev_assignments = np.full(n, -1, dtype=int)
        for _ in range(max_iter):
            distances = np.sum(
                (features[:, None, :] - centers[None, :, :]) ** 2, axis=2
            )
            assignments = np.argmin(distances, axis=1)
            if np.array_equal(assignments, prev_assignments):
                break
            prev_assignments = assignments
            for j in range(k):
                members = features[assignments == j]
                if len(members) > 0:
                    centers[j] = members.mean(axis=0)
        inertia = float(
            np.sum((features - centers[assignments]) ** 2)
        )
        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers.copy()
            best_assignments = assignments.copy()
    assert best_centers is not None and best_assignments is not None
    return Cover(centers=best_centers, train_assignments=best_assignments)


def assign_nearest(features: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distances = np.sum(
        (features[:, None, :] - centers[None, :, :]) ** 2, axis=2
    )
    return np.argmin(distances, axis=1)


def build_nerve(
    cover: Cover,
    stream_features: np.ndarray,
    overlap_radius: float,
) -> SheafEdges:
    """Edge (i, j) included iff at least one stream context lies within
    `overlap_radius` of both c_i and c_j (Euclidean)."""
    K = cover.K
    overlap_counts: dict[tuple[int, int], int] = {}
    overlap_means: dict[tuple[int, int], np.ndarray] = {}
    # Per-vertex membership mask
    distances = np.sqrt(
        np.sum(
            (stream_features[:, None, :] - cover.centers[None, :, :]) ** 2,
            axis=2,
        )
    )  # (N_train, K)
    membership = distances <= overlap_radius  # (N_train, K)
    for i in range(K):
        for j in range(i + 1, K):
            mask = membership[:, i] & membership[:, j]
            count = int(mask.sum())
            if count == 0:
                continue
            overlap_counts[(i, j)] = count
            overlap_means[(i, j)] = stream_features[mask].mean(axis=0)
    edges = tuple(sorted(overlap_counts.keys()))
    return SheafEdges(
        edges=edges,
        overlap_counts=overlap_counts,
        overlap_means=overlap_means,
    )


def fit_restrictions(
    cover: Cover,
    nerve: SheafEdges,
    ridge: float = 0.10,
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """For each edge (i, j), fit rho_i and rho_j in R^{D x D} so that
    rho_i @ c_i ~ m_{ij} ~ rho_j @ c_j, with ridge regularisation toward
    the identity:
        rho_k* = argmin ||rho c_k - m||^2 + ridge ||rho - I||_F^2.

    Closed form via the per-row solution. Identical regularizer for each
    row, so solve once: rho* = (m c_k^T + ridge I) (c_k c_k^T + ridge I)^{-1}.
    """
    D = cover.D
    restrictions: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    eye = np.eye(D)
    for (i, j) in nerve.edges:
        m = nerve.overlap_means[(i, j)]
        c_i = cover.centers[i]
        c_j = cover.centers[j]
        gram_i = np.outer(c_i, c_i) + ridge * eye
        gram_j = np.outer(c_j, c_j) + ridge * eye
        rhs_i = np.outer(m, c_i) + ridge * eye
        rhs_j = np.outer(m, c_j) + ridge * eye
        rho_i = np.linalg.solve(gram_i.T, rhs_i.T).T
        rho_j = np.linalg.solve(gram_j.T, rhs_j.T).T
        restrictions[(i, j)] = (rho_i, rho_j)
    return restrictions


def assemble_coboundary(
    cover: Cover,
    nerve: SheafEdges,
    restrictions: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Block matrix delta_0 of shape (E*D, K*D)."""
    K, D, E = cover.K, cover.D, nerve.E
    delta = np.zeros((E * D, K * D), dtype=float)
    for e_idx, (i, j) in enumerate(nerve.edges):
        rho_i, rho_j = restrictions[(i, j)]
        delta[e_idx * D : (e_idx + 1) * D, i * D : (i + 1) * D] = rho_i
        delta[e_idx * D : (e_idx + 1) * D, j * D : (j + 1) * D] = -rho_j
    return delta


def cohomology_dims(delta_0: np.ndarray, K: int, E: int, D: int, tol: float = 1e-9) -> tuple[int, int, int]:
    if delta_0.size == 0:
        return K * D, 0, 0
    rank = int(np.linalg.matrix_rank(delta_0, tol=tol))
    dim_h0 = max(K * D - rank, 0)
    dim_h1 = max(E * D - rank, 0)
    return dim_h0, dim_h1, rank


def coboundary_energy(
    sections: np.ndarray,
    nerve: SheafEdges,
    restrictions: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
) -> float:
    """sum_{ij} ||rho_i sigma_i - rho_j sigma_j||^2 (weighted by overlap count)."""
    if not nerve.edges:
        return 0.0
    energy = 0.0
    for (i, j) in nerve.edges:
        rho_i, rho_j = restrictions[(i, j)]
        residual = rho_i @ sections[i] - rho_j @ sections[j]
        weight = float(nerve.overlap_counts[(i, j)])
        energy += weight * float(np.dot(residual, residual))
    return energy


def glue_sections(
    cover: Cover,
    nerve: SheafEdges,
    restrictions: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    gamma: float = 1.0,
) -> np.ndarray:
    """Solve argmin_sigma   sum_{ij} count_{ij} ||rho_i sigma_i - m_{ij}||^2
                          + sum_{ij} count_{ij} ||rho_j sigma_j - m_{ij}||^2
                          + gamma * sum_i ||sigma_i - c_i||^2.

    Linear in sigma. Solve K independent D x D systems per vertex
    (the cross-vertex term in the loss vanishes because each edge term
    decouples into one term in sigma_i and one in sigma_j).
    """
    K, D = cover.K, cover.D
    glued = cover.centers.copy()
    for i in range(K):
        gram = gamma * np.eye(D)
        rhs = gamma * cover.centers[i]
        for (a, b) in nerve.edges:
            if a == i:
                rho_a, _ = restrictions[(a, b)]
                count = float(nerve.overlap_counts[(a, b)])
                m = nerve.overlap_means[(a, b)]
                gram += count * rho_a.T @ rho_a
                rhs += count * rho_a.T @ m
            elif b == i:
                _, rho_b = restrictions[(a, b)]
                count = float(nerve.overlap_counts[(a, b)])
                m = nerve.overlap_means[(a, b)]
                gram += count * rho_b.T @ rho_b
                rhs += count * rho_b.T @ m
        glued[i] = np.linalg.solve(gram, rhs)
    return glued


def fit_sheaf(
    cover: Cover,
    stream_features: np.ndarray,
    overlap_radius: float,
    restriction_ridge: float = 0.10,
    glue_gamma: float = 1.0,
) -> SheafFit:
    nerve = build_nerve(cover, stream_features, overlap_radius)
    restrictions = fit_restrictions(cover, nerve, ridge=restriction_ridge)
    delta_0 = assemble_coboundary(cover, nerve, restrictions)
    dim_h0, dim_h1, rank = cohomology_dims(delta_0, cover.K, nerve.E, cover.D)
    energy_initial = coboundary_energy(cover.centers, nerve, restrictions)
    glued = glue_sections(cover, nerve, restrictions, gamma=glue_gamma)
    energy_after = coboundary_energy(glued, nerve, restrictions)
    sheaf_laplacian = delta_0.T @ delta_0 if delta_0.size else np.zeros((cover.K * cover.D, cover.K * cover.D))
    return SheafFit(
        restrictions=restrictions,
        delta_0=delta_0,
        sheaf_laplacian=sheaf_laplacian,
        rank_delta=rank,
        dim_h0=dim_h0,
        dim_h1=dim_h1,
        coboundary_energy_initial=energy_initial,
        coboundary_energy_after_glue=energy_after,
        glued_centers=glued,
    )


def best_action_from_section(
    section: np.ndarray,
    direct_actions: Iterable[str],
    unsafe_weight: float,
) -> str:
    """Action utility = success - unsafe_weight * unsafe. Sections are
    laid out as [a1_success, a1_unsafe, a2_success, a2_unsafe, ...]."""
    utilities: list[tuple[float, str]] = []
    for index, action in enumerate(direct_actions):
        success = float(section[2 * index])
        unsafe = float(section[2 * index + 1])
        utilities.append((success - unsafe_weight * unsafe, action))
    utilities.sort(key=lambda item: (-item[0], item[1]))
    return utilities[0][1]
