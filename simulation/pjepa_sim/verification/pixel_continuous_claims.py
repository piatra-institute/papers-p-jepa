"""Executable checks for the pixel continuous-control benchmark."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

from pjepa_sim.paths import OUTPUT_DIR
from pjepa_sim.perception.continuous import run_pixel_continuous_benchmark


OUT = OUTPUT_DIR / "pixel_continuous_verification.json"


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
    results = run_pixel_continuous_benchmark()
    learners = results["learners"]
    active = learners["pixel_active_probe"]
    no_probe = learners["pixel_no_probe"]
    entropy = learners["pixel_entropy_probe"]
    oracle = learners["oracle_regime"]
    claims = [
        Claim(
            name="pixel_active_probe_beats_no_probe_score",
            passed=active["risk_adjusted_score"] > no_probe["risk_adjusted_score"] + 0.04,
            observed=round(active["risk_adjusted_score"] - no_probe["risk_adjusted_score"], 6),
            threshold="> 0.04 score margin",
            detail="Active probing should improve pixel-based continuous control when rendered observations alias hidden dynamics.",
        ),
        Claim(
            name="pixel_active_probe_reduces_unsafe_failure",
            passed=no_probe["unsafe_failure_rate"] - active["unsafe_failure_rate"] > 0.005,
            observed=round(no_probe["unsafe_failure_rate"] - active["unsafe_failure_rate"], 6),
            threshold="> 0.005 unsafe reduction",
            detail="Safe probes should reduce unsafe continuous-control actions.",
        ),
        Claim(
            name="pixel_active_probe_is_entropy_competitive",
            passed=active["risk_adjusted_score"] > entropy["risk_adjusted_score"] - 0.03,
            observed=round(active["risk_adjusted_score"] - entropy["risk_adjusted_score"], 6),
            threshold="> -0.03 score margin",
            detail="Value-aware probing should remain competitive with entropy probing from rendered pixels.",
        ),
        Claim(
            name="pixel_active_probe_uses_probes",
            passed=active["mean_probes"] > 0.10,
            observed=round(active["mean_probes"], 6),
            threshold="> 0.10 probes",
            detail="The policy should actually gather information before committing to a controller.",
        ),
        Claim(
            name="pixel_active_probe_recovers_oracle_value",
            passed=oracle["risk_adjusted_score"] - active["risk_adjusted_score"] < 0.50,
            observed=round(oracle["risk_adjusted_score"] - active["risk_adjusted_score"], 6),
            threshold="< 0.50 score gap",
            detail="Learned pixel-based probing should recover some hidden-regime oracle value, while leaving substantial headroom.",
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
