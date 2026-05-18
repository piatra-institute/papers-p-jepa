"""H4 - Sheaf framing is decorative.

Hypothesis: a real cellular sheaf with learned linear restriction maps
and a ||delta sigma||^2 loss does not beat the scalar
posterior-weighted-variance baseline on dishworld. (The existing
`representation/gluing.py` is not this experiment - its sections are
handed in, not learned from a cover; there is no nerve, no Laplacian,
no H^0/H^1.)

Pass criterion (preregistered):
  - mean(sheaf_score - scalar_score) < 0.02 across seeds, AND
  - 95% paired bootstrap CI on per-seed delta contains zero.

Setup:
  - Generate train stream and test contexts via `generate_records`
    (visual labels shifted between train and test).
  - Build a K-cluster cover by deterministic k-means on the train
    stream's action_features. K is set above the true regime count so
    that intra-regime clusters fragment and have overlapping support -
    otherwise the cover has no edges and the sheaf is trivially equal
    to the scalar baseline.
  - Build the 1-skeleton, fit linear restriction maps, assemble the
    coboundary operator, compute dim H^0 and dim H^1, and glue the
    vertex sections.
  - Score raw centers (scalar baseline) vs glued centers (sheaf model)
    on held-out test contexts. Both score by mapping each test context
    to its nearest train-cover center, reading off success-vs-unsafe
    from the section vector, and taking the best utility-weighted action.
  - Repeat across 20 seeds with different stream shuffles.

Run:
    cd simulation
    uv run python -m pjepa_sim.experiments.h4_sheaf_vs_scalar
"""

from __future__ import annotations

import json
import sys
from typing import Any

import numpy as np

from pjepa_sim.core.dishworld import ACTION_MODEL, DIRECT_ACTIONS, REGIMES
from pjepa_sim.experiments.bootstrap import paired_bootstrap_ci
from pjepa_sim.paths import OUTPUT_DIR
from pjepa_sim.representation.learning import generate_records
from pjepa_sim.representation.sheaf_toy import (
    assign_nearest,
    best_action_from_section,
    fit_sheaf,
    kmeans_cover,
)


N_SEEDS = 20
BASE_SEED = 41
CONTEXTS_PER_REGIME = 96
ACTION_FEATURE_NOISE = 0.035
K_CLUSTERS = 6  # > 4 regimes so intra-regime fragments overlap
RADIUS_PERCENTILE = 65.0  # percentile of nearest-center distances to set overlap radius
UNSAFE_WEIGHT = 2.0
SCORE_MEAN_TOLERANCE = 0.02


def score_centers(
    test_records,
    centers: np.ndarray,
    train_centers: np.ndarray,
) -> tuple[float, float, float]:
    """Map each test record to nearest train cluster, choose best action
    from that cluster's section, evaluate against the true regime.
    Returns (success_rate, unsafe_rate, risk_adjusted_score).
    """
    test_features = np.asarray([record.action_features for record in test_records], dtype=float)
    assignments = assign_nearest(test_features, train_centers)
    total_success = 0.0
    total_unsafe = 0.0
    for record, cluster in zip(test_records, assignments):
        section = centers[int(cluster)]
        action = best_action_from_section(section, DIRECT_ACTIONS, UNSAFE_WEIGHT)
        outcome = ACTION_MODEL[record.regime][action]
        total_success += outcome.success
        total_unsafe += outcome.unsafe
    n = float(len(test_records))
    success_rate = total_success / n
    unsafe_rate = total_unsafe / n
    score = success_rate - UNSAFE_WEIGHT * unsafe_rate
    return success_rate, unsafe_rate, score


def cluster_purity(records, assignments: np.ndarray) -> float:
    total = 0
    correct = 0
    for cluster in sorted(set(int(a) for a in assignments)):
        labels = [
            record.regime
            for record, a in zip(records, assignments)
            if int(a) == cluster
        ]
        if not labels:
            continue
        counts = {label: labels.count(label) for label in set(labels)}
        correct += max(counts.values())
        total += len(labels)
    return float(correct / total) if total else 0.0


def run_single_seed(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    train_records = generate_records("train", CONTEXTS_PER_REGIME, ACTION_FEATURE_NOISE, rng)
    test_records = generate_records("test", CONTEXTS_PER_REGIME, ACTION_FEATURE_NOISE, rng)

    train_features = np.asarray([record.action_features for record in train_records], dtype=float)

    cover = kmeans_cover(train_features, k=K_CLUSTERS, seed=seed)

    # Set overlap radius from data: percentile of distance to second-nearest center.
    dists = np.sqrt(
        np.sum((train_features[:, None, :] - cover.centers[None, :, :]) ** 2, axis=2)
    )  # (N, K)
    second_nearest = np.partition(dists, kth=1, axis=1)[:, 1]
    overlap_radius = float(np.percentile(second_nearest, RADIUS_PERCENTILE))

    fit = fit_sheaf(
        cover=cover,
        stream_features=train_features,
        overlap_radius=overlap_radius,
        restriction_ridge=0.10,
        glue_gamma=1.0,
    )

    scalar_success, scalar_unsafe, scalar_score = score_centers(
        test_records, cover.centers, cover.centers
    )
    sheaf_success, sheaf_unsafe, sheaf_score = score_centers(
        test_records, fit.glued_centers, cover.centers
    )

    train_assignments = cover.train_assignments
    train_purity = cluster_purity(train_records, train_assignments)
    test_assignments = assign_nearest(
        np.asarray([record.action_features for record in test_records], dtype=float),
        cover.centers,
    )
    test_purity = cluster_purity(test_records, test_assignments)

    return {
        "seed": seed,
        "k": cover.K,
        "overlap_radius": overlap_radius,
        "n_edges": fit.delta_0.shape[0] // cover.D if fit.delta_0.size else 0,
        "rank_delta": fit.rank_delta,
        "dim_h0": fit.dim_h0,
        "dim_h1": fit.dim_h1,
        "coboundary_energy_initial": fit.coboundary_energy_initial,
        "coboundary_energy_after_glue": fit.coboundary_energy_after_glue,
        "train_cluster_purity": train_purity,
        "test_cluster_purity": test_purity,
        "scalar_success_rate": scalar_success,
        "scalar_unsafe_rate": scalar_unsafe,
        "scalar_score": scalar_score,
        "sheaf_success_rate": sheaf_success,
        "sheaf_unsafe_rate": sheaf_unsafe,
        "sheaf_score": sheaf_score,
        "sheaf_minus_scalar_score": sheaf_score - scalar_score,
        "sheaf_minus_scalar_unsafe": sheaf_unsafe - scalar_unsafe,
    }


def run() -> dict[str, Any]:
    seeds = [BASE_SEED + offset for offset in range(N_SEEDS)]
    per_seed: list[dict[str, Any]] = []
    score_deltas: list[float] = []
    unsafe_deltas: list[float] = []
    for seed in seeds:
        record = run_single_seed(seed)
        per_seed.append(record)
        score_deltas.append(float(record["sheaf_minus_scalar_score"]))
        unsafe_deltas.append(float(record["sheaf_minus_scalar_unsafe"]))

    score_ci = paired_bootstrap_ci(score_deltas, resamples=10_000, seed=BASE_SEED + 1)
    unsafe_ci = paired_bootstrap_ci(unsafe_deltas, resamples=10_000, seed=BASE_SEED + 2)

    mean_scalar_score = float(np.mean([r["scalar_score"] for r in per_seed]))
    mean_sheaf_score = float(np.mean([r["sheaf_score"] for r in per_seed]))
    mean_initial_energy = float(np.mean([r["coboundary_energy_initial"] for r in per_seed]))
    mean_glued_energy = float(np.mean([r["coboundary_energy_after_glue"] for r in per_seed]))
    mean_n_edges = float(np.mean([r["n_edges"] for r in per_seed]))
    mean_dim_h0 = float(np.mean([r["dim_h0"] for r in per_seed]))
    mean_dim_h1 = float(np.mean([r["dim_h1"] for r in per_seed]))

    pass_score = (score_ci.mean < SCORE_MEAN_TOLERANCE) and score_ci.contains_zero()
    verdict = "PASS" if pass_score else "FAIL"

    result = {
        "experiment": "H4_sheaf_vs_scalar",
        "hypothesis": (
            "A real cellular sheaf with learned linear restriction maps and a "
            "||delta sigma||^2 gluing loss does not outperform the raw cover "
            "centers (scalar baseline) on downstream action choice."
        ),
        "pass_criterion": (
            f"mean(sheaf_score - scalar_score) < {SCORE_MEAN_TOLERANCE} AND "
            f"95% paired bootstrap CI on per-seed delta contains zero "
            f"across {N_SEEDS} seeds"
        ),
        "preregistered": True,
        "config": {
            "n_seeds": N_SEEDS,
            "base_seed": BASE_SEED,
            "contexts_per_regime": CONTEXTS_PER_REGIME,
            "action_feature_noise": ACTION_FEATURE_NOISE,
            "k_clusters": K_CLUSTERS,
            "true_regime_count": len(REGIMES),
            "radius_percentile": RADIUS_PERCENTILE,
            "unsafe_weight": UNSAFE_WEIGHT,
            "score_mean_tolerance": SCORE_MEAN_TOLERANCE,
        },
        "summary": {
            "mean_scalar_score": mean_scalar_score,
            "mean_sheaf_score": mean_sheaf_score,
            "mean_sheaf_minus_scalar_score": score_ci.mean,
            "sheaf_minus_scalar_score_ci": score_ci.as_dict(),
            "mean_sheaf_minus_scalar_unsafe": unsafe_ci.mean,
            "sheaf_minus_scalar_unsafe_ci": unsafe_ci.as_dict(),
            "mean_n_edges": mean_n_edges,
            "mean_dim_h0": mean_dim_h0,
            "mean_dim_h1": mean_dim_h1,
            "mean_coboundary_energy_initial": mean_initial_energy,
            "mean_coboundary_energy_after_glue": mean_glued_energy,
            "energy_reduction_factor": (
                mean_initial_energy / mean_glued_energy
                if mean_glued_energy > 0 else float("inf")
            ),
        },
        "per_seed": per_seed,
        "verdict": verdict,
        "interpretation": (
            "If PASS: the sheaf machinery - learned restrictions, coboundary, "
            "Laplacian, H^0 / H^1 dimensions - does not produce better action "
            "choice than the raw cluster centers on dishworld. The framing in "
            "the paper title and abstract is decorative on this benchmark. The "
            "reported coboundary energy reduction (which is non-trivial) "
            "improves intra-edge consistency without translating into downstream "
            "task value. If FAIL: the sheaf gluing is doing real work; report "
            "the per-seed delta CI and the dim H^1 numbers in the paper."
        ),
        "next_actions": (
            "If PASS: in a follow-up session, demote the sheaf framing from the "
            "title to a motivation paragraph; keep sheaf_toy.py as a "
            "negative-result artifact with the verifier and the JSON. "
            "If FAIL: promote sheaf_toy.py to a load-bearing module and replace "
            "the placeholder gluing benchmark (gluing.py) with this construction."
        ),
    }

    out_dir = OUTPUT_DIR / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "h4_sheaf_vs_scalar.json"
    with out_path.open("w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)

    print(
        f"[H4] {verdict}: sheaf-scalar score delta mean={score_ci.mean:+.4f} "
        f"CI95=[{score_ci.low:+.4f},{score_ci.high:+.4f}] "
        f"(scalar={mean_scalar_score:.3f} sheaf={mean_sheaf_score:.3f})"
    )
    print(
        f"[H4] mean K={K_CLUSTERS} edges={mean_n_edges:.1f} "
        f"dim H^0={mean_dim_h0:.1f} dim H^1={mean_dim_h1:.1f} "
        f"energy: {mean_initial_energy:.4f} -> {mean_glued_energy:.4f}"
    )
    print(f"[H4] wrote {out_path}")
    return result


def main() -> int:
    result = run()
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
