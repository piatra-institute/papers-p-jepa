"""Small deterministic clustering helpers for representation benchmarks."""

from __future__ import annotations

import numpy as np


def standardise(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-8] = 1.0
    return (features - mean) / std, mean, std


def assign_nearest(features: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distances = np.asarray([np.sum((features - center) ** 2, axis=1) for center in centers])
    return np.argmin(distances, axis=0)


def kmeans(features: np.ndarray, k: int, seed: int, iterations: int = 80) -> tuple[np.ndarray, np.ndarray]:
    return kmeans_once(features, k, seed, iterations)


def best_kmeans(
    features: np.ndarray,
    k: int,
    seed: int,
    restarts: int = 24,
    iterations: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    best_assignments: np.ndarray | None = None
    best_centers: np.ndarray | None = None
    best_inertia = float("inf")
    for offset in range(restarts):
        assignments, centers = kmeans_once(features, k, seed + offset, iterations)
        inertia = float(
            sum(
                np.sum((features[index] - centers[int(assignments[index])]) ** 2)
                for index in range(len(features))
            )
        )
        if inertia < best_inertia:
            best_assignments = assignments
            best_centers = centers
            best_inertia = inertia
    if best_assignments is None or best_centers is None:
        raise RuntimeError("k-means failed to produce a clustering")
    return best_assignments, best_centers


def kmeans_once(
    features: np.ndarray,
    k: int,
    seed: int,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, len(features)))
    centers = [features[first]]
    while len(centers) < k:
        distances = np.min(
            np.asarray([np.sum((features - center) ** 2, axis=1) for center in centers]),
            axis=0,
        )
        total = float(distances.sum())
        if total <= 0.0:
            centers.append(features[len(centers) % len(features)])
        else:
            centers.append(features[int(rng.choice(len(features), p=distances / total))])
    centers_array = np.asarray(centers, dtype=float)
    assignments = assign_nearest(features, centers_array)
    for _ in range(iterations):
        next_centers = centers_array.copy()
        for cluster in range(k):
            selected = features[assignments == cluster]
            if len(selected):
                next_centers[cluster] = selected.mean(axis=0)
        next_assignments = assign_nearest(features, next_centers)
        if np.array_equal(next_assignments, assignments):
            return next_assignments, next_centers
        assignments = next_assignments
        centers_array = next_centers
    return assignments, centers_array
