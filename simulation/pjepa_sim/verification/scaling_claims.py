"""Executable checks for the synthetic representation-scaling benchmark."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

from pjepa_sim.paths import OUTPUT_DIR
from pjepa_sim.representation.scaling import run_scaling_benchmark


OUT = OUTPUT_DIR / "scaling_verification.json"


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
    results = run_scaling_benchmark()
    cases = results["cases"]
    action_margins_vs_appearance = []
    action_margins_vs_prior = []
    oracle_gaps = []
    action_purities = []
    action_scores = []

    for case in cases.values():
        learners = case["learners"]
        action = learners["action_consequence_grouping"]
        appearance = learners["appearance_grouping"]
        prior = learners["prior_average"]
        oracle = learners["oracle_regime"]
        action_margins_vs_appearance.append(action["risk_adjusted_score"] - appearance["risk_adjusted_score"])
        action_margins_vs_prior.append(action["risk_adjusted_score"] - prior["risk_adjusted_score"])
        oracle_gaps.append(oracle["risk_adjusted_score"] - action["risk_adjusted_score"])
        action_purities.append(action["cluster_purity"])
        action_scores.append(action["risk_adjusted_score"])

    claims = [
        Claim(
            name="scaling_action_consequence_beats_appearance_all_counts",
            passed=min(action_margins_vs_appearance) > 0.25,
            observed=round(min(action_margins_vs_appearance), 6),
            threshold="minimum margin > 0.25",
            detail="Action-consequence grouping should beat visual grouping at every tested regime count.",
        ),
        Claim(
            name="scaling_action_consequence_beats_prior_all_counts",
            passed=min(action_margins_vs_prior) > 0.25,
            observed=round(min(action_margins_vs_prior), 6),
            threshold="minimum margin > 0.25",
            detail="Local action sections should beat one global average section at every tested regime count.",
        ),
        Claim(
            name="scaling_action_consequence_keeps_high_purity",
            passed=min(action_purities) > 0.90,
            observed=round(min(action_purities), 6),
            threshold="minimum purity > 0.90",
            detail="The learned cover should remain aligned with hidden action regimes across the synthetic scale sweep.",
        ),
        Claim(
            name="scaling_action_consequence_approaches_oracle",
            passed=max(oracle_gaps) < 0.08,
            observed=round(max(oracle_gaps), 6),
            threshold="maximum oracle gap < 0.08",
            detail="Action-consequence grouping should recover most oracle action value across the sweep.",
        ),
        Claim(
            name="scaling_largest_case_remains_useful",
            passed=action_scores[-1] > 0.70,
            observed=round(action_scores[-1], 6),
            threshold="32-regime score > 0.70",
            detail="The largest synthetic case should remain a useful representation benchmark rather than collapse.",
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
