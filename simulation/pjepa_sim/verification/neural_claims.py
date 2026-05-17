"""Executable checks for neural intervention encoder claims."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

from pjepa_sim.paths import OUTPUT_DIR
from pjepa_sim.representation.neural import run_neural_benchmark


OUT = OUTPUT_DIR / "neural_verification.json"


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
    results = run_neural_benchmark()
    learners = results["learners"]
    neural = learners["neural_p_representation"]
    appearance = learners["appearance_only_encoder"]
    prior = learners["prior_average"]
    engineered = learners["engineered_fingerprint_reference"]

    claims = [
        Claim(
            name="neural_p_representation_beats_appearance_score",
            passed=neural["risk_adjusted_score"] > appearance["risk_adjusted_score"] + 0.20,
            observed=round(neural["risk_adjusted_score"] - appearance["risk_adjusted_score"], 6),
            threshold="> 0.20 score margin",
            detail="A learned intervention-conditioned representation should beat visual grouping under visual shift.",
        ),
        Claim(
            name="neural_p_representation_beats_prior_score",
            passed=neural["risk_adjusted_score"] > prior["risk_adjusted_score"] + 0.10,
            observed=round(neural["risk_adjusted_score"] - prior["risk_adjusted_score"], 6),
            threshold="> 0.10 score margin",
            detail="The learned local representation should beat one global average action model.",
        ),
        Claim(
            name="neural_p_representation_has_high_purity",
            passed=neural["cluster_purity"] > 0.85,
            observed=round(neural["cluster_purity"], 6),
            threshold="> 0.85",
            detail="The learned predicted-test vector should recover hidden action regimes without labels as features.",
        ),
        Claim(
            name="neural_p_representation_approaches_engineered_reference",
            passed=engineered["risk_adjusted_score"] - neural["risk_adjusted_score"] < 0.12,
            observed=round(engineered["risk_adjusted_score"] - neural["risk_adjusted_score"], 6),
            threshold="< 0.12 score gap",
            detail="The learned representation should approximate the engineered action-consequence fingerprint.",
        ),
        Claim(
            name="hidden_labels_not_used_as_features",
            passed=not bool(results["config"]["hidden_labels_used_as_features"]),
            observed=float(bool(results["config"]["hidden_labels_used_as_features"])),
            threshold="= 0",
            detail="Hidden regime labels may be used only for diagnostics and evaluation.",
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
