"""H2 - Active-probing advantage over entropy is seed noise.

Hypothesis: the five-seed sweep at neural_active.py:677 shows active
probing beats entropy by mean 0.005 with one seed favouring entropy. With
more seeds and a real paired bootstrap CI, the active-vs-entropy delta
contains zero.

Pass criterion (preregistered): 95% paired bootstrap CI on per-seed
(active - entropy) risk-adjusted-score delta contains zero.

Also reports the no-probe margin and unsafe-failure reduction across the
extended sweep, since those were the claims the original sweep treated as
robust.

Run:
    cd simulation
    uv run python -m pjepa_sim.experiments.h2_seed_sweep_bootstrap
"""

from __future__ import annotations

import json
import sys
from typing import Any

import numpy as np

from pjepa_sim.experiments.bootstrap import paired_bootstrap_ci
from pjepa_sim.paths import OUTPUT_DIR
from pjepa_sim.representation.neural_active import run_neural_active_seed_sweep_benchmark


SEEDS = tuple(range(53, 103))  # 50 deterministic seeds
RESAMPLES = 10_000


def run() -> dict[str, Any]:
    sweep = run_neural_active_seed_sweep_benchmark(seeds=SEEDS)
    cases = sweep["cases"]

    active_minus_entropy: list[float] = []
    active_minus_no_probe: list[float] = []
    no_probe_unsafe_minus_active_unsafe: list[float] = []
    active_scores: list[float] = []
    entropy_scores: list[float] = []
    no_probe_scores: list[float] = []
    oracle_gaps: list[float] = []

    for seed in SEEDS:
        case = cases[str(seed)]
        summary = case["summary"]
        active_minus_entropy.append(float(summary["active_minus_entropy_score"]))
        active_minus_no_probe.append(float(summary["active_minus_no_probe_score"]))
        no_probe_unsafe_minus_active_unsafe.append(float(summary["no_probe_minus_active_unsafe"]))
        active_scores.append(float(summary["active_score"]))
        entropy_scores.append(float(case["learners"]["learned_entropy_probe"]["risk_adjusted_score"]))
        no_probe_scores.append(float(case["learners"]["learned_no_probe"]["risk_adjusted_score"]))
        oracle_gaps.append(float(summary["oracle_gap"]))

    entropy_ci = paired_bootstrap_ci(active_minus_entropy, resamples=RESAMPLES, seed=11)
    no_probe_ci = paired_bootstrap_ci(active_minus_no_probe, resamples=RESAMPLES, seed=13)
    unsafe_ci = paired_bootstrap_ci(no_probe_unsafe_minus_active_unsafe, resamples=RESAMPLES, seed=17)

    n_seeds_entropy_wins = sum(1 for d in active_minus_entropy if d < 0)
    n_seeds_active_wins = sum(1 for d in active_minus_entropy if d > 0)
    n_seeds_tie = len(SEEDS) - n_seeds_entropy_wins - n_seeds_active_wins

    pass_entropy_null = entropy_ci.contains_zero()
    verdict = "PASS" if pass_entropy_null else "FAIL"

    result = {
        "experiment": "H2_seed_sweep_bootstrap",
        "hypothesis": (
            "The active-probing advantage over entropy probing in the learned "
            "active-probing seed sweep is consistent with seed noise; the 95% "
            "paired bootstrap CI on (active - entropy) contains zero."
        ),
        "pass_criterion": (
            f"95% paired bootstrap CI on per-seed (active - entropy) "
            f"risk_adjusted_score delta contains zero across {len(SEEDS)} seeds"
        ),
        "preregistered": True,
        "config": {
            "n_seeds": len(SEEDS),
            "seeds": list(SEEDS),
            "resamples": RESAMPLES,
        },
        "summary": {
            "mean_active_score": float(np.mean(active_scores)),
            "std_active_score": float(np.std(active_scores)),
            "mean_entropy_score": float(np.mean(entropy_scores)),
            "mean_no_probe_score": float(np.mean(no_probe_scores)),
            "mean_active_minus_entropy": entropy_ci.mean,
            "active_minus_entropy_ci": entropy_ci.as_dict(),
            "mean_active_minus_no_probe": no_probe_ci.mean,
            "active_minus_no_probe_ci": no_probe_ci.as_dict(),
            "mean_no_probe_unsafe_minus_active_unsafe": unsafe_ci.mean,
            "no_probe_unsafe_minus_active_unsafe_ci": unsafe_ci.as_dict(),
            "mean_oracle_gap": float(np.mean(oracle_gaps)),
            "n_seeds_active_wins_vs_entropy": n_seeds_active_wins,
            "n_seeds_entropy_wins_vs_active": n_seeds_entropy_wins,
            "n_seeds_tied": n_seeds_tie,
        },
        "verdict": verdict,
        "interpretation": (
            "If PASS: the value-aware vs entropy comparison from the 5-seed sweep "
            "is consistent with noise; the 'value-aware probing beats entropy' "
            "claim is not supported by this study. The no-probe margin and unsafe "
            "reduction CIs should still be reported - those are different claims. "
            "If FAIL: the active-vs-entropy delta survives 50 seeds; the original "
            "5-seed sentence in the abstract should be replaced with the bootstrap CI."
        ),
        "next_actions": (
            "If PASS: retract the 'value-aware probing beats entropy' claim in a "
            "follow-up session; keep the no-probe-margin and unsafe-reduction "
            "claims if their CIs exclude zero (they likely will - the original "
            "sweep already showed every seed favoured active over no-probe). "
            "If FAIL: replace the 5-seed prose with the bootstrap CI in the paper."
        ),
    }

    out_dir = OUTPUT_DIR / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "h2_seed_sweep_bootstrap.json"
    with out_path.open("w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)

    print(
        f"[H2] {verdict}: active-entropy delta mean={entropy_ci.mean:+.4f} "
        f"CI95=[{entropy_ci.low:+.4f},{entropy_ci.high:+.4f}] "
        f"(n_active_wins={n_seeds_active_wins}/{len(SEEDS)} "
        f"n_entropy_wins={n_seeds_entropy_wins}/{len(SEEDS)} ties={n_seeds_tie})"
    )
    print(
        f"[H2] active-no_probe delta mean={no_probe_ci.mean:+.4f} "
        f"CI95=[{no_probe_ci.low:+.4f},{no_probe_ci.high:+.4f}] contains_zero={no_probe_ci.contains_zero()}"
    )
    print(
        f"[H2] no_probe_unsafe-active_unsafe delta mean={unsafe_ci.mean:+.4f} "
        f"CI95=[{unsafe_ci.low:+.4f},{unsafe_ci.high:+.4f}] contains_zero={unsafe_ci.contains_zero()}"
    )
    print(f"[H2] wrote {out_path}")
    return result


def main() -> int:
    result = run()
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
