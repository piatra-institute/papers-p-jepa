"""H5 - JEPA augmentations from embodied/causal mathematics improve
downstream action choice on dishworld.

Hypothesis (per-augmentation): at least one of the augmentations
{intervention, bisimulation, active masking, viability head, all
combined} produces a non-zero paired bootstrap CI advantage in
risk-adjusted score over base JEPA on dishworld.

This is a *toy* test. A positive result here is directional evidence
that the augmentation is worth implementing at V-JEPA scale (see
docs/JEPA_AUGMENTATIONS.md for the PyTorch specifications). A negative
result deprioritises that augmentation.

Setup:
  - Base JEPA: encoder + EMA target + mask predictor, trained with
    mask-fill loss only, random masking.
  - +intervention: adds outcome_predictor and L_do per epoch.
  - +bisim: adds L_bisim per epoch.
  - +active: replaces random masking with hard-example mining.
  - +viability: adds viability_head trained against true unsafe rates.
  - +all: all of the above enabled jointly.
  - Eval: cluster latents (k=4) by k-means; assign test contexts;
    choose best utility-weighted action per cluster; report risk-adjusted
    score on test set.

Run:
    cd simulation
    uv run python -m pjepa_sim.experiments.h5_jepa_augmentations
"""

from __future__ import annotations

import json
import sys
from typing import Any

import numpy as np

from pjepa_sim.experiments.bootstrap import paired_bootstrap_ci
from pjepa_sim.jepa_toy.data import generate_contexts
from pjepa_sim.jepa_toy.eval import (
    evaluate_action_choice,
    evaluate_viability_action_choice,
)
from pjepa_sim.jepa_toy.training import TrainingConfig, train_jepa_toy
from pjepa_sim.paths import OUTPUT_DIR


N_SEEDS = 12
BASE_SEED = 101
N_EPOCHS = 500


def base_config() -> TrainingConfig:
    return TrainingConfig(n_epochs=N_EPOCHS)


def variants() -> dict[str, TrainingConfig]:
    cfg = lambda **kw: TrainingConfig(n_epochs=N_EPOCHS, **kw)
    return {
        "base_jepa": cfg(),
        "+intervention": cfg(enable_intervention=True),
        "+bisim": cfg(enable_bisimulation=True),
        "+active_mask": cfg(enable_active_masking=True),
        "+viability": cfg(enable_viability=True),
        "+all": cfg(
            enable_intervention=True,
            enable_bisimulation=True,
            enable_active_masking=True,
            enable_viability=True,
        ),
    }


def evaluate_one_seed(name: str, cfg: TrainingConfig, seed: int) -> dict[str, float]:
    result = train_jepa_toy(cfg, seed=seed)
    rng = np.random.default_rng(seed + 1000)
    train = generate_contexts("train", cfg.contexts_per_regime, cfg.sensor_noise, rng)
    test = generate_contexts("test", cfg.contexts_per_regime, cfg.sensor_noise, rng)
    metrics = evaluate_action_choice(result.model, train, test, cluster_seed=seed + 2000)
    extras: dict[str, float] = {}
    if cfg.enable_viability:
        extras = evaluate_viability_action_choice(result.model, test)
    return {
        "variant": name,
        "seed": seed,
        **metrics,
        **extras,
    }


def run() -> dict[str, Any]:
    seeds = [BASE_SEED + i for i in range(N_SEEDS)]
    all_records: list[dict[str, Any]] = []
    by_variant: dict[str, list[dict[str, Any]]] = {}

    for seed in seeds:
        for name, cfg in variants().items():
            record = evaluate_one_seed(name, cfg, seed)
            all_records.append(record)
            by_variant.setdefault(name, []).append(record)
            print(
                f"  seed={seed} {name}: "
                f"score={record['risk_adjusted_score']:+.4f} "
                f"test_purity={record['test_cluster_purity']:.3f}"
            )

    base_scores = [r["risk_adjusted_score"] for r in by_variant["base_jepa"]]

    augmentation_summary: dict[str, Any] = {}
    for name, records in by_variant.items():
        scores = [r["risk_adjusted_score"] for r in records]
        purities = [r["test_cluster_purity"] for r in records]
        unsafes = [r["unsafe_failure_rate"] for r in records]
        deltas = [s - b for s, b in zip(scores, base_scores)]
        summary: dict[str, Any] = {
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "mean_test_purity": float(np.mean(purities)),
            "mean_unsafe_failure_rate": float(np.mean(unsafes)),
            "n_seeds": len(records),
        }
        if name != "base_jepa":
            ci = paired_bootstrap_ci(deltas, resamples=10_000, seed=seed_for(name))
            summary["mean_minus_base"] = float(np.mean(deltas))
            summary["minus_base_ci"] = ci.as_dict()
            summary["beats_base"] = (not ci.contains_zero()) and ci.mean > 0
        augmentation_summary[name] = summary

    pass_any = any(
        s.get("beats_base", False)
        for name, s in augmentation_summary.items()
        if name != "base_jepa"
    )
    verdict = "PASS" if pass_any else "FAIL"

    winners = [name for name, s in augmentation_summary.items() if s.get("beats_base", False)]
    losers = [
        name
        for name, s in augmentation_summary.items()
        if name != "base_jepa" and not s.get("beats_base", False)
    ]

    result = {
        "experiment": "H5_jepa_augmentations",
        "hypothesis": (
            "At least one of {intervention, bisim, active masking, viability, "
            "all} produces a non-zero paired bootstrap CI advantage over base "
            "JEPA on the dishworld toy."
        ),
        "pass_criterion": (
            f"At least one augmentation has 95% CI on (variant - base) "
            f"per-seed delta strictly above zero across {N_SEEDS} seeds."
        ),
        "preregistered": True,
        "config": {
            "n_seeds": N_SEEDS,
            "base_seed": BASE_SEED,
            "n_epochs": N_EPOCHS,
            "variants": list(variants().keys()),
        },
        "summary_by_variant": augmentation_summary,
        "all_records": all_records,
        "verdict": verdict,
        "winners": winners,
        "losers": losers,
        "interpretation": (
            "If PASS: the named augmentation(s) produce a real directional "
            "advantage in the toy. This is *not* a real V-JEPA result but it "
            "is positive evidence that the augmentation is worth GPU time. "
            "The corresponding section of docs/JEPA_AUGMENTATIONS.md is "
            "promoted from speculative to priority. "
            "If FAIL: no augmentation moves the needle on this toy. Either the "
            "toy is too small to reveal the effect, or the augmentation's "
            "inductive bias is not appropriate for action-choice on linearly-"
            "separable Bernoulli sources. Recommendation: try the augmentation "
            "on a setting with the structure the augmentation is designed for "
            "(e.g. intervention loss needs partial-observation context to "
            "show its strength; sheaf loss needs continuous-overlap data)."
        ),
        "next_actions": (
            "Promote winners in docs/JEPA_AUGMENTATIONS.md to 'priority for "
            "V-JEPA implementation'. Document losers and the likely reason. "
            "Note that toy negatives are weak evidence: a small dishworld may "
            "not exercise the augmentation's true target."
        ),
    }

    out_dir = OUTPUT_DIR / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "h5_jepa_augmentations.json"
    with out_path.open("w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)

    print()
    print(f"[H5] {verdict}: winners={winners} losers={losers}")
    for name, s in augmentation_summary.items():
        if name == "base_jepa":
            print(f"  {name}: score={s['mean_score']:+.4f} purity={s['mean_test_purity']:.3f}")
        else:
            ci = s["minus_base_ci"]
            print(
                f"  {name}: score={s['mean_score']:+.4f} purity={s['mean_test_purity']:.3f} "
                f"delta vs base: mean={ci['mean']:+.4f} CI95=[{ci['low']:+.4f},{ci['high']:+.4f}] "
                f"beats_base={s['beats_base']}"
            )
    print(f"[H5] wrote {out_path}")
    return result


def seed_for(name: str) -> int:
    return abs(hash(name)) % 2**30


def main() -> int:
    result = run()
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
