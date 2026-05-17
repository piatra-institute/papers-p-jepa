"""Executable checks for action-grounded representation learning."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

from pjepa_sim.paths import OUTPUT_DIR
from pjepa_sim.representation.learning import run_representation_benchmark


OUT = OUTPUT_DIR / "representation_verification.json"


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
    results = run_representation_benchmark()
    learners = results["learners"]
    action = learners["action_consequence_grouping"]
    appearance = learners["appearance_grouping"]
    prior = learners["prior_average"]
    oracle = learners["oracle_regime"]

    claims = [
        Claim(
            name="action_consequence_representation_beats_appearance_score",
            passed=action["risk_adjusted_score"] > appearance["risk_adjusted_score"] + 0.25,
            observed=round(action["risk_adjusted_score"] - appearance["risk_adjusted_score"], 6),
            threshold="> 0.25",
            detail="Action-grounded grouping should beat visual grouping under visual cue shift.",
        ),
        Claim(
            name="action_consequence_representation_beats_prior_score",
            passed=action["risk_adjusted_score"] > prior["risk_adjusted_score"] + 0.10,
            observed=round(action["risk_adjusted_score"] - prior["risk_adjusted_score"], 6),
            threshold="> 0.10",
            detail="Learned local representations should beat one global average action model.",
        ),
        Claim(
            name="action_consequence_clusters_have_high_regime_purity",
            passed=action["cluster_purity"] > 0.95,
            observed=round(action["cluster_purity"], 6),
            threshold="> 0.95",
            detail="The learner should recover action regimes from fingerprints without regime labels.",
        ),
        Claim(
            name="action_consequence_approaches_oracle_score",
            passed=oracle["risk_adjusted_score"] - action["risk_adjusted_score"] < 0.08,
            observed=round(oracle["risk_adjusted_score"] - action["risk_adjusted_score"], 6),
            threshold="< 0.08 gap",
            detail="Action-grounded representation should recover most oracle regime value.",
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
