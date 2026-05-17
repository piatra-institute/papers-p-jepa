"""Executable checks for learned active-probing seed robustness."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

from pjepa_sim.paths import OUTPUT_DIR
from pjepa_sim.representation.neural_active import run_neural_active_seed_sweep_benchmark


OUT = OUTPUT_DIR / "neural_active_seed_sweep_verification.json"


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
    results = run_neural_active_seed_sweep_benchmark()
    summary = results["summary"]
    claims = [
        Claim(
            name="active_probe_beats_no_probe_on_average",
            passed=summary["active_minus_no_probe_mean"] > 0.18,
            observed=round(summary["active_minus_no_probe_mean"], 6),
            threshold="> 0.18 mean score margin",
            detail="Across seeds, active probing should clearly beat acting from the aliased initial observation.",
        ),
        Claim(
            name="active_probe_beats_no_probe_every_seed",
            passed=summary["active_minus_no_probe_min"] > 0.12,
            observed=round(summary["active_minus_no_probe_min"], 6),
            threshold="> 0.12 minimum score margin",
            detail="The active-probing gain should not depend on one favorable seed.",
        ),
        Claim(
            name="active_probe_reduces_unsafe_on_average",
            passed=summary["no_probe_minus_active_unsafe_mean"] > 0.05,
            observed=round(summary["no_probe_minus_active_unsafe_mean"], 6),
            threshold="> 0.05 mean unsafe reduction",
            detail="Across seeds, active probing should reduce unsafe failure.",
        ),
        Claim(
            name="active_probe_reduces_unsafe_every_seed",
            passed=summary["no_probe_minus_active_unsafe_min"] > 0.03,
            observed=round(summary["no_probe_minus_active_unsafe_min"], 6),
            threshold="> 0.03 minimum unsafe reduction",
            detail="The safety gain should not depend on one favorable seed.",
        ),
        Claim(
            name="active_probe_is_entropy_competitive_on_average",
            passed=summary["active_minus_entropy_mean"] > 0.0,
            observed=round(summary["active_minus_entropy_mean"], 6),
            threshold="> 0.0 mean score margin",
            detail="Value-aware probing should be at least competitive with entropy probing on average across seeds.",
        ),
        Claim(
            name="active_probe_has_no_large_entropy_regression",
            passed=summary["active_minus_entropy_min"] > -0.02,
            observed=round(summary["active_minus_entropy_min"], 6),
            threshold="> -0.02 minimum score margin",
            detail="A seed sweep may contain entropy-favorable runs, but active probing should not suffer a large regression.",
        ),
        Claim(
            name="active_probe_stays_near_oracle_on_average",
            passed=summary["oracle_gap_mean"] < 0.13,
            observed=round(summary["oracle_gap_mean"], 6),
            threshold="< 0.13 mean oracle gap",
            detail="Learned probing should recover much of the hidden-regime oracle value across seeds.",
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
        "summary": summary,
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
