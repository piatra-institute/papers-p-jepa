"""Executable checks for skill composition from learned representations."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

from pjepa_sim.paths import OUTPUT_DIR
from pjepa_sim.representation.composition import run_skill_composition_benchmark


OUT = OUTPUT_DIR / "skill_composition_verification.json"


@dataclass(frozen=True)
class Claim:
    name: str
    passed: bool
    observed: float | str
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
    results = run_skill_composition_benchmark()
    learners = results["learners"]
    action = learners["action_consequence_grouping"]
    appearance = learners["appearance_grouping"]
    prior = learners["prior_average"]
    oracle = learners["oracle_regime"]
    expected_chains = {
        "dry": "no_prep->fast_lift",
        "soapy": "wipe->two_contact_lift",
        "cracked": "cushion->slow_lift",
        "heavy": "brace->grip_hard",
    }
    actual_chains = {
        regime: action["by_regime"][regime]["dominant_chain"]
        for regime in expected_chains
    }

    claims = [
        Claim(
            name="composition_representation_beats_appearance_score",
            passed=action["risk_adjusted_score"] > appearance["risk_adjusted_score"] + 0.30,
            observed=round(action["risk_adjusted_score"] - appearance["risk_adjusted_score"], 6),
            threshold="> 0.30",
            detail="Action-grounded composition should beat visual grouping under visual cue shift.",
        ),
        Claim(
            name="composition_representation_beats_prior_score",
            passed=action["risk_adjusted_score"] > prior["risk_adjusted_score"] + 0.10,
            observed=round(action["risk_adjusted_score"] - prior["risk_adjusted_score"], 6),
            threshold="> 0.10",
            detail="Local composition sections should beat a single global chain model.",
        ),
        Claim(
            name="composition_clusters_have_high_regime_purity",
            passed=action["cluster_purity"] > 0.95,
            observed=round(action["cluster_purity"], 6),
            threshold="> 0.95",
            detail="Composition fingerprints should recover action regimes without regime labels.",
        ),
        Claim(
            name="composition_representation_approaches_oracle_score",
            passed=oracle["risk_adjusted_score"] - action["risk_adjusted_score"] < 0.08,
            observed=round(oracle["risk_adjusted_score"] - action["risk_adjusted_score"], 6),
            threshold="< 0.08 gap",
            detail="Learned composition representation should recover most oracle regime value.",
        ),
        Claim(
            name="composition_selects_expected_skill_chains",
            passed=actual_chains == expected_chains,
            observed=", ".join(f"{key}:{value}" for key, value in actual_chains.items()),
            threshold=", ".join(f"{key}:{value}" for key, value in expected_chains.items()),
            detail="The learned representation should compose the intended prepare and finish skills.",
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
