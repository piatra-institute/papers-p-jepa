"""Executable checks for online cover construction."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

from pjepa_sim.paths import OUTPUT_DIR
from pjepa_sim.representation.online import run_online_cover_benchmark


OUT = OUTPUT_DIR / "online_cover_verification.json"


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
    results = run_online_cover_benchmark()
    learners = results["learners"]
    online = learners["online_action_cover"]
    appearance = learners["appearance_online"]
    prior = learners["prior_average"]
    oracle = learners["oracle_regime"]

    claims = [
        Claim(
            name="online_cover_discovers_four_action_regimes",
            passed=online["n_clusters"] == 4,
            observed=float(online["n_clusters"]),
            threshold="= 4",
            detail="Incremental cover construction should discover the four action regimes.",
        ),
        Claim(
            name="online_cover_has_high_regime_purity",
            passed=online["cluster_purity"] > 0.95,
            observed=round(online["cluster_purity"], 6),
            threshold="> 0.95",
            detail="The discovered online cover should align with hidden action regimes.",
        ),
        Claim(
            name="online_cover_beats_appearance_score",
            passed=online["risk_adjusted_score"] > appearance["risk_adjusted_score"] + 0.25,
            observed=round(online["risk_adjusted_score"] - appearance["risk_adjusted_score"], 6),
            threshold="> 0.25",
            detail="Online action-consequence cover should beat visual grouping under visual shift.",
        ),
        Claim(
            name="online_cover_beats_prior_score",
            passed=online["risk_adjusted_score"] > prior["risk_adjusted_score"] + 0.10,
            observed=round(online["risk_adjusted_score"] - prior["risk_adjusted_score"], 6),
            threshold="> 0.10",
            detail="Online local covers should beat one global average action model.",
        ),
        Claim(
            name="online_cover_approaches_oracle_score",
            passed=oracle["risk_adjusted_score"] - online["risk_adjusted_score"] < 0.08,
            observed=round(oracle["risk_adjusted_score"] - online["risk_adjusted_score"], 6),
            threshold="< 0.08 gap",
            detail="Online cover construction should recover most oracle action value.",
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
