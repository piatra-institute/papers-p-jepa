"""Executable checks for learned neural active-probing claims."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

from pjepa_sim.paths import OUTPUT_DIR
from pjepa_sim.representation.neural_active import run_neural_active_probe_benchmark


OUT = OUTPUT_DIR / "neural_active_probe_verification.json"


@dataclass(frozen=True)
class Claim:
    name: str
    passed: bool
    observed: float
    threshold: str
    detail: str

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "observed": self.observed,
            "threshold": self.threshold,
            "detail": self.detail,
        }


def main() -> None:
    results = run_neural_active_probe_benchmark()
    learners = results["learners"]
    active = learners["learned_active_probe"]
    no_probe = learners["learned_no_probe"]
    entropy = learners["learned_entropy_probe"]
    oracle = learners["oracle_regime"]
    claims = [
        Claim(
            name="learned_active_probe_beats_no_probe_score",
            passed=active["risk_adjusted_score"] > no_probe["risk_adjusted_score"] + 0.18,
            observed=round(active["risk_adjusted_score"] - no_probe["risk_adjusted_score"], 6),
            threshold="> 0.18 score margin",
            detail="Learned active probing should improve decisions when initial sensors alias hidden regimes.",
        ),
        Claim(
            name="learned_active_probe_reduces_unsafe_failure",
            passed=no_probe["unsafe_failure_rate"] - active["unsafe_failure_rate"] > 0.06,
            observed=round(no_probe["unsafe_failure_rate"] - active["unsafe_failure_rate"], 6),
            threshold="> 0.06 unsafe reduction",
            detail="Safe probes should reduce unsafe failures relative to acting from the ambiguous initial observation.",
        ),
        Claim(
            name="learned_active_probe_beats_entropy_probe_score",
            passed=active["risk_adjusted_score"] > entropy["risk_adjusted_score"] + 0.015,
            observed=round(active["risk_adjusted_score"] - entropy["risk_adjusted_score"], 6),
            threshold="> 0.015 score margin",
            detail="Value-aware probe choice should beat probe selection by predicted evidence entropy.",
        ),
        Claim(
            name="learned_active_probe_uses_probes",
            passed=active["mean_probes"] > 0.5,
            observed=round(active["mean_probes"], 6),
            threshold="> 0.5 probes",
            detail="The active policy should actually gather information before acting.",
        ),
        Claim(
            name="learned_active_probe_approaches_oracle",
            passed=oracle["risk_adjusted_score"] - active["risk_adjusted_score"] < 0.12,
            observed=round(oracle["risk_adjusted_score"] - active["risk_adjusted_score"], 6),
            threshold="< 0.12 score gap",
            detail="Learned probing should recover much of the hidden-regime oracle value.",
        ),
        Claim(
            name="hidden_labels_not_used_as_features",
            passed=not bool(results["config"]["hidden_labels_used_as_features"]),
            observed=float(bool(results["config"]["hidden_labels_used_as_features"])),
            threshold="= 0",
            detail="Hidden regime labels may be used only for diagnostics and evaluation.",
        ),
    ]
    report = {
        "passed": all(claim.passed for claim in claims),
        "num_claims": len(claims),
        "num_passed": sum(1 for claim in claims if claim.passed),
        "claims": [claim.as_dict() for claim in claims],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")

    for claim in claims:
        status = "PASS" if claim.passed else "FAIL"
        print(f"{status}  {claim.name}")
        print(f"      observed: {claim.observed}")
        print(f"      required: {claim.threshold}")
    print()
    print(f"Wrote: {OUT}")

    if not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
