"""Executable checks for restriction-map gluing claims."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

from pjepa_sim.paths import OUTPUT_DIR
from pjepa_sim.representation.gluing import run_gluing_ablation_benchmark


OUT = OUTPUT_DIR / "gluing_ablation_verification.json"


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
    results = run_gluing_ablation_benchmark()
    learners = results["learners"]
    no_glue = learners["identity_no_glue"]
    learned = learners["learned_restriction_glue"]
    reference = learners["reference_only"]
    oracle_restriction = learners["oracle_restriction_glue"]
    oracle_regime = learners["oracle_regime"]

    claims = [
        Claim(
            name="learned_restrictions_reduce_overlap_residual",
            passed=learned["mean_overlap_residual"] < no_glue["mean_overlap_residual"] * 0.10,
            observed=round(learned["mean_overlap_residual"] / no_glue["mean_overlap_residual"], 6),
            threshold="< 0.10 of identity/no-glue residual",
            detail="Learned restriction maps should make overlapping local action sections agree.",
        ),
        Claim(
            name="learned_restrictions_beat_identity_no_glue_score",
            passed=learned["risk_adjusted_score"] > no_glue["risk_adjusted_score"] + 0.15,
            observed=round(learned["risk_adjusted_score"] - no_glue["risk_adjusted_score"], 6),
            threshold="> 0.15 score margin",
            detail="The same local sections should act better after their action-coordinate interfaces are glued.",
        ),
        Claim(
            name="learned_restrictions_beat_reference_only_score",
            passed=learned["risk_adjusted_score"] > reference["risk_adjusted_score"] + 0.005,
            observed=round(learned["risk_adjusted_score"] - reference["risk_adjusted_score"], 6),
            threshold="> 0.005 score margin",
            detail="Gluing multiple local sections should improve over the noisy reference section alone.",
        ),
        Claim(
            name="learned_restrictions_approach_oracle_restrictions",
            passed=abs(oracle_restriction["risk_adjusted_score"] - learned["risk_adjusted_score"]) < 0.02,
            observed=round(abs(oracle_restriction["risk_adjusted_score"] - learned["risk_adjusted_score"]), 6),
            threshold="< 0.02 score gap",
            detail="Restriction maps learned from overlaps should approach the hand-coded coordinate maps.",
        ),
        Claim(
            name="learned_restrictions_approach_oracle_regime",
            passed=oracle_regime["risk_adjusted_score"] - learned["risk_adjusted_score"] < 0.02,
            observed=round(oracle_regime["risk_adjusted_score"] - learned["risk_adjusted_score"], 6),
            threshold="< 0.02 score gap",
            detail="The glued local sections should recover most of the hidden-regime oracle value in this toy setup.",
        ),
    ]

    summary = {
        "passed": all(claim.passed for claim in claims),
        "num_claims": len(claims),
        "num_passed": sum(1 for claim in claims if claim.passed),
        "claims": [claim.as_dict() for claim in claims],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2) + "\n")

    for claim in claims:
        status = "PASS" if claim.passed else "FAIL"
        print(f"{status}  {claim.name}")
        print(f"      observed: {claim.observed}")
        print(f"      required: {claim.threshold}")
    print()
    print(f"Wrote: {OUT}")

    if not summary["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
