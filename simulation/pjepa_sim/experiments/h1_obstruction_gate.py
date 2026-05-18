"""H1 - Obstruction gate is a no-op.

Hypothesis: `p_jepa_stack` and `active_psr_probe` produce identical numeric
output on every configured suite, because the gate inside
`_belief_decision_metrics` (suites.py:383) never fires below
`spec.sheaf_threshold` for any non-terminal posterior reached on the
existing suites.

Pass criterion: max absolute difference in `risk_adjusted_score` (and any
other reported scalar metric) across all five suites is < 1e-9.

The two agents in `evaluate_suite` are literally the same
`_decision_probe_result` function called with `use_obstruction_gate=True`
vs `False`. Comparing the public outputs is sufficient: if they tie
numerically the gate has no operational role on these suites.

Run:
    cd simulation
    uv run python -m pjepa_sim.experiments.h1_obstruction_gate
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from pjepa_sim.benchmark.suites import available_suites, evaluate_suite, load_spec
from pjepa_sim.paths import OUTPUT_DIR


METRICS = (
    "risk_adjusted_score",
    "success_rate",
    "unsafe_failure_rate",
    "mean_probes",
    "mean_obstruction_at_action",
)
PASS_TOLERANCE = 1e-9


def run() -> dict[str, Any]:
    suite_names = available_suites()
    per_suite: dict[str, dict[str, Any]] = {}
    max_abs_delta = 0.0
    max_delta_metric = ""
    max_delta_suite = ""

    for name in suite_names:
        spec = load_spec(name)
        report = evaluate_suite(spec, ["p_jepa_stack", "active_psr_probe"])
        stack = report["agents"]["p_jepa_stack"]
        psr = report["agents"]["active_psr_probe"]
        deltas: dict[str, float] = {}
        for metric in METRICS:
            delta = float(stack[metric]) - float(psr[metric])
            deltas[metric] = delta
            if abs(delta) > max_abs_delta:
                max_abs_delta = abs(delta)
                max_delta_metric = metric
                max_delta_suite = name
        per_suite[name] = {
            "sheaf_threshold": spec.sheaf_threshold,
            "obstruction_at_policy_prior": report["obstruction"]["policy_prior"],
            "obstruction_above_threshold": (
                report["obstruction"]["policy_prior"] > spec.sheaf_threshold
            ),
            "p_jepa_stack": {metric: float(stack[metric]) for metric in METRICS},
            "active_psr_probe": {metric: float(psr[metric]) for metric in METRICS},
            "deltas": deltas,
        }

    verdict = "PASS" if max_abs_delta < PASS_TOLERANCE else "FAIL"
    result = {
        "experiment": "H1_obstruction_gate",
        "hypothesis": (
            "p_jepa_stack and active_psr_probe produce identical numeric "
            "output on every configured suite (gate is a no-op)."
        ),
        "pass_criterion": f"max |delta| across {METRICS} and all suites is < {PASS_TOLERANCE}",
        "preregistered": True,
        "suites": list(suite_names),
        "max_abs_delta": max_abs_delta,
        "max_delta_metric": max_delta_metric,
        "max_delta_suite": max_delta_suite,
        "verdict": verdict,
        "per_suite": per_suite,
        "interpretation": (
            "If PASS: the obstruction gate has no operational effect on the existing "
            "suites. Either the policy-prior obstruction (0.255) already lies above "
            "every suite's sheaf_threshold so the gate never short-circuits a probe, "
            "or every short-circuited decision matches what the no-gate policy chose "
            "anyway. The p_jepa_stack name is operationally redundant with "
            "active_psr_probe on these configurations. If FAIL: identify the suite "
            "and metric where the gate changes behaviour."
        ),
        "next_actions": (
            "If PASS: in a follow-up session, merge p_jepa_stack and active_psr_probe "
            "in the paper, or add a suite with sheaf_threshold > prior obstruction to "
            "actually exercise the gate. If FAIL: document the exercising suite in "
            "the paper and keep both agent names."
        ),
    }

    out_dir = OUTPUT_DIR / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "h1_obstruction_gate.json"
    with out_path.open("w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)

    print(f"[H1] {verdict}: max |delta| = {max_abs_delta:.3e} "
          f"(metric={max_delta_metric!r} suite={max_delta_suite!r})")
    print(f"[H1] wrote {out_path}")
    return result


def main() -> int:
    result = run()
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
