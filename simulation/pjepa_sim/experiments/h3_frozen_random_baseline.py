"""H3 - Frozen random projection matches the trained MLP.

Hypothesis: the "neural P-representation" recovers the four hidden regimes
because the action-consequence outputs of the four Bernoulli sources are
already nearly linearly separable in sensor space, not because the MLP is
learning anything specific. A frozen random projection of equal width
should match the trained MLP on risk-adjusted score and cluster purity.

Pass criterion (preregistered):
  - mean(trained_score - frozen_score) < 0.05 across seeds, AND
  - 95% paired bootstrap CI on the per-seed delta contains zero.

The runner re-uses `evaluate_neural_p_representation` with two encoders:
 1. a frozen `TinyMLP.create(...)` instance (never `.fit()`-ed)
 2. the standard trained `train_model(...)` instance
on identical train/test contexts and intervention records per seed.

Run:
    cd simulation
    uv run python -m pjepa_sim.experiments.h3_frozen_random_baseline
"""

from __future__ import annotations

import json
import sys
from typing import Any

import numpy as np

from pjepa_sim.experiments.bootstrap import paired_bootstrap_ci
from pjepa_sim.paths import OUTPUT_DIR
from pjepa_sim.representation.neural import (
    SENSOR_BASE,
    TESTS,
    TinyMLP,
    evaluate_appearance,
    evaluate_neural_p_representation,
    fit_appearance_sections,
    generate_contexts,
    generate_interventions,
    train_model,
)


N_SEEDS = 10
BASE_SEED = 29
CONTEXTS_PER_REGIME = 128
INTERVENTION_REPEATS = 10
SENSOR_NOISE = 0.045
NUISANCE_DIM = 4
HIDDEN_DIM = 32
UNSAFE_WEIGHT = 2.0
SCORE_MEAN_TOLERANCE = 0.05


def build_untrained_mlp(input_dim: int, hidden_dim: int, output_dim: int, seed: int) -> TinyMLP:
    """`TinyMLP.create` with the same shapes/seed as `train_model` would
    use, but `.fit()` is never called. Weights remain at He-init values."""
    return TinyMLP.create(input_dim, hidden_dim, output_dim, seed)


def run_single_seed(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    train_contexts = generate_contexts("train", CONTEXTS_PER_REGIME, SENSOR_NOISE, NUISANCE_DIM, rng)
    test_contexts = generate_contexts("test", CONTEXTS_PER_REGIME, SENSOR_NOISE, NUISANCE_DIM, rng)
    train_records = generate_interventions(train_contexts, INTERVENTION_REPEATS, rng)

    sensor_feature_dim = len(SENSOR_BASE["dry"])
    input_dim = sensor_feature_dim + len(TESTS)
    output_dim = 2

    feature_getter = lambda context: context.sensor_features  # noqa: E731

    appearance_sections, _ = fit_appearance_sections(train_contexts, train_records)
    appearance = evaluate_appearance(test_contexts, appearance_sections, UNSAFE_WEIGHT)

    frozen_model = build_untrained_mlp(input_dim, HIDDEN_DIM, output_dim, seed + 1)
    frozen = evaluate_neural_p_representation(
        train_contexts,
        test_contexts,
        train_records,
        frozen_model,
        feature_getter,
        UNSAFE_WEIGHT,
        seed,
    )

    trained_model = train_model(
        train_contexts,
        train_records,
        feature_getter=feature_getter,
        hidden_dim=HIDDEN_DIM,
        seed=seed + 1,
    )
    trained = evaluate_neural_p_representation(
        train_contexts,
        test_contexts,
        train_records,
        trained_model,
        feature_getter,
        UNSAFE_WEIGHT,
        seed,
    )

    return {
        "seed": seed,
        "appearance": appearance.as_dict(),
        "frozen_random": frozen.as_dict(),
        "trained": trained.as_dict(),
        "score_delta_trained_minus_frozen": (
            trained.risk_adjusted_score - frozen.risk_adjusted_score
        ),
        "score_delta_trained_minus_appearance": (
            trained.risk_adjusted_score - appearance.risk_adjusted_score
        ),
        "purity_delta_trained_minus_frozen": (
            trained.cluster_purity - frozen.cluster_purity
        ),
        "prediction_error_delta_trained_minus_frozen": (
            trained.mean_prediction_error - frozen.mean_prediction_error
        ),
    }


def run() -> dict[str, Any]:
    seeds = [BASE_SEED + offset for offset in range(N_SEEDS)]
    per_seed: list[dict[str, Any]] = []
    score_deltas: list[float] = []
    purity_deltas: list[float] = []
    for seed in seeds:
        record = run_single_seed(seed)
        per_seed.append(record)
        score_deltas.append(float(record["score_delta_trained_minus_frozen"]))
        purity_deltas.append(float(record["purity_delta_trained_minus_frozen"]))

    score_ci = paired_bootstrap_ci(score_deltas, resamples=10_000, seed=BASE_SEED)
    purity_ci = paired_bootstrap_ci(purity_deltas, resamples=10_000, seed=BASE_SEED + 1)

    mean_trained = float(np.mean([r["trained"]["risk_adjusted_score"] for r in per_seed]))
    mean_frozen = float(np.mean([r["frozen_random"]["risk_adjusted_score"] for r in per_seed]))
    mean_appearance = float(np.mean([r["appearance"]["risk_adjusted_score"] for r in per_seed]))
    mean_trained_purity = float(np.mean([r["trained"]["cluster_purity"] for r in per_seed]))
    mean_frozen_purity = float(np.mean([r["frozen_random"]["cluster_purity"] for r in per_seed]))

    pass_score = (score_ci.mean < SCORE_MEAN_TOLERANCE) and score_ci.contains_zero()
    verdict = "PASS" if pass_score else "FAIL"

    result = {
        "experiment": "H3_frozen_random_baseline",
        "hypothesis": (
            "A frozen random TinyMLP of the same width matches the trained MLP "
            "on risk-adjusted score for the neural P-representation."
        ),
        "pass_criterion": (
            "mean(trained_score - frozen_score) < 0.05 AND 95% paired bootstrap CI "
            "on per-seed delta contains zero"
        ),
        "preregistered": True,
        "config": {
            "n_seeds": N_SEEDS,
            "base_seed": BASE_SEED,
            "contexts_per_regime": CONTEXTS_PER_REGIME,
            "intervention_repeats": INTERVENTION_REPEATS,
            "sensor_noise": SENSOR_NOISE,
            "nuisance_dim": NUISANCE_DIM,
            "hidden_dim": HIDDEN_DIM,
            "unsafe_weight": UNSAFE_WEIGHT,
            "score_mean_tolerance": SCORE_MEAN_TOLERANCE,
        },
        "summary": {
            "mean_appearance_score": mean_appearance,
            "mean_frozen_random_score": mean_frozen,
            "mean_trained_score": mean_trained,
            "mean_trained_minus_frozen_score": score_ci.mean,
            "trained_minus_frozen_score_ci": score_ci.as_dict(),
            "mean_frozen_random_purity": mean_frozen_purity,
            "mean_trained_purity": mean_trained_purity,
            "mean_trained_minus_frozen_purity": purity_ci.mean,
            "trained_minus_frozen_purity_ci": purity_ci.as_dict(),
        },
        "per_seed": per_seed,
        "verdict": verdict,
        "interpretation": (
            "If PASS: the trained MLP does not meaningfully outperform a frozen "
            "random projection of equal width on the 4-regime structured-sensor task. "
            "The 'neural intervention encoder' label oversells what is essentially "
            "test-vector clustering on separable Bernoulli sources. "
            "If FAIL: the MLP is doing real work; report the score delta and the "
            "purity delta in the paper, with the frozen-random column kept as a "
            "stronger baseline."
        ),
        "next_actions": (
            "If PASS: in a follow-up session, demote the 'neural P-representation' "
            "framing to 'test-vector clustering' and add the frozen-random column to "
            "the relevant paper tables. If FAIL: add the frozen-random column anyway "
            "as a stronger baseline, but keep the 'neural intervention encoder' name."
        ),
    }

    out_dir = OUTPUT_DIR / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "h3_frozen_random_baseline.json"
    with out_path.open("w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)

    print(f"[H3] {verdict}: trained-frozen score delta mean={score_ci.mean:+.4f} "
          f"CI95=[{score_ci.low:+.4f},{score_ci.high:+.4f}] "
          f"(appearance={mean_appearance:.3f} frozen={mean_frozen:.3f} trained={mean_trained:.3f})")
    print(f"[H3] purity delta mean={purity_ci.mean:+.4f} "
          f"CI95=[{purity_ci.low:+.4f},{purity_ci.high:+.4f}] "
          f"(frozen={mean_frozen_purity:.3f} trained={mean_trained_purity:.3f})")
    print(f"[H3] wrote {out_path}")
    return result


def main() -> int:
    result = run()
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
