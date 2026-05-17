"""Executable checks for neural intervention sample-efficiency claims."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

from pjepa_sim.paths import OUTPUT_DIR
from pjepa_sim.representation.neural import run_neural_sample_efficiency_benchmark


OUT = OUTPUT_DIR / "neural_sample_efficiency_verification.json"


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
    results = run_neural_sample_efficiency_benchmark()
    summary = results["summary"]
    claims = [
        Claim(
            name="sample_efficiency_neural_beats_prior_all_counts",
            passed=summary["min_neural_minus_prior"] > 0.25,
            observed=round(summary["min_neural_minus_prior"], 6),
            threshold="minimum margin > 0.25",
            detail="The learned predicted-test representation should beat the prior baseline at every tested intervention budget.",
        ),
        Claim(
            name="sample_efficiency_neural_beats_appearance_all_counts",
            passed=summary["min_neural_minus_appearance"] > 0.45,
            observed=round(summary["min_neural_minus_appearance"], 6),
            threshold="minimum margin > 0.45",
            detail="The learned predicted-test representation should beat visual grouping at every tested intervention budget.",
        ),
        Claim(
            name="sample_efficiency_neural_keeps_high_purity",
            passed=summary["min_neural_purity"] > 0.95,
            observed=round(summary["min_neural_purity"], 6),
            threshold="minimum purity > 0.95",
            detail="The learned predicted-test vectors should keep regime-separating structure even with sparse samples.",
        ),
        Claim(
            name="sample_efficiency_neural_approaches_engineered_reference",
            passed=summary["max_engineered_gap"] < 0.02,
            observed=round(summary["max_engineered_gap"], 6),
            threshold="maximum score gap < 0.02",
            detail="The learned representation should approach the engineered fingerprint reference across tested budgets.",
        ),
        Claim(
            name="sample_efficiency_prediction_error_decreases",
            passed=summary["prediction_error_reduction"] > 0.02,
            observed=round(summary["prediction_error_reduction"], 6),
            threshold="first-last MAE reduction > 0.02",
            detail="More intervention repeats should improve the learned predicted-test estimates.",
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
