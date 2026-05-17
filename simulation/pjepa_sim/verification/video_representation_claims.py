"""Executable checks for the video-representation surrogate benchmark."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from dataclasses import replace

import numpy as np

import pjepa_sim.perception.video_representation as video
from pjepa_sim.paths import OUTPUT_DIR
from pjepa_sim.perception.video_representation import run_video_representation_benchmark


OUT = OUTPUT_DIR / "video_representation_verification.json"


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
    results = run_video_representation_benchmark()
    learners = results["learners"]
    passive = learners["jepa_passive_video"]
    action = learners["p_action_representation"]
    prior = learners["prior_average"]
    oracle = learners["oracle_regime"]
    random_action = _random_action_evidence_control()
    permuted_action = _permuted_test_action_evidence_control()
    repeat_one = run_video_representation_benchmark(intervention_repeats=1)["learners"]["p_action_representation"]
    repeat_sixteen = run_video_representation_benchmark(intervention_repeats=16)["learners"]["p_action_representation"]
    claims = [
        Claim(
            name="p_action_representation_beats_passive_video_score",
            passed=action["risk_adjusted_score"] > passive["risk_adjusted_score"] + 0.30,
            observed=round(action["risk_adjusted_score"] - passive["risk_adjusted_score"], 6),
            threshold="> 0.30 score margin",
            detail="Action-conditioned predicted-test representations should beat passive video prediction under visual shift.",
        ),
        Claim(
            name="p_action_representation_beats_prior_score",
            passed=action["risk_adjusted_score"] > prior["risk_adjusted_score"] + 0.20,
            observed=round(action["risk_adjusted_score"] - prior["risk_adjusted_score"], 6),
            threshold="> 0.20 score margin",
            detail="The action-conditioned representation should beat one global prior action model.",
        ),
        Claim(
            name="passive_video_predicts_future_frames",
            passed=passive["passive_prediction_mae"] < 0.04,
            observed=round(passive["passive_prediction_mae"], 6),
            threshold="< 0.04 MAE",
            detail="The passive JEPA-like surrogate should be competent at its own next-frame objective.",
        ),
        Claim(
            name="passive_video_has_low_action_regime_purity",
            passed=passive["cluster_purity"] < 0.70,
            observed=round(passive["cluster_purity"], 6),
            threshold="< 0.70 purity",
            detail="Good passive prediction should not imply action-regime recovery under the constructed visual shift.",
        ),
        Claim(
            name="p_action_representation_has_high_regime_purity",
            passed=action["cluster_purity"] > 0.90,
            observed=round(action["cluster_purity"], 6),
            threshold="> 0.90 purity",
            detail="The action-conditioned representation should recover hidden action regimes without label features.",
        ),
        Claim(
            name="p_action_representation_approaches_oracle",
            passed=oracle["risk_adjusted_score"] - action["risk_adjusted_score"] < 0.08,
            observed=round(oracle["risk_adjusted_score"] - action["risk_adjusted_score"], 6),
            threshold="< 0.08 score gap",
            detail="The action-conditioned representation should recover most hidden-regime oracle value.",
        ),
        Claim(
            name="hidden_labels_not_used_as_features",
            passed=not bool(results["config"]["hidden_labels_used_as_features"]),
            observed=float(bool(results["config"]["hidden_labels_used_as_features"])),
            threshold="= 0",
            detail="Hidden regime labels may be used only for diagnostics and evaluation.",
        ),
        Claim(
            name="actual_v_jepa_not_claimed",
            passed=not bool(results["config"]["actual_v_jepa_or_video_foundation_model_run"]),
            observed=float(bool(results["config"]["actual_v_jepa_or_video_foundation_model_run"])),
            threshold="= 0",
            detail="This benchmark is a local surrogate and must not be reported as an actual V-JEPA comparison.",
        ),
        Claim(
            name="p_action_representation_depends_on_action_evidence",
            passed=action["risk_adjusted_score"] > random_action.risk_adjusted_score + 0.35,
            observed=round(action["risk_adjusted_score"] - random_action.risk_adjusted_score, 6),
            threshold="> 0.35 score margin",
            detail="Replacing intervention evidence with random features should destroy most of the P-representation advantage.",
        ),
        Claim(
            name="permuting_test_action_evidence_breaks_transfer",
            passed=action["risk_adjusted_score"] > permuted_action.risk_adjusted_score + 0.50,
            observed=round(action["risk_adjusted_score"] - permuted_action.risk_adjusted_score, 6),
            threshold="> 0.50 score margin",
            detail="The representation should depend on matched context-to-intervention evidence, not just the marginal feature distribution.",
        ),
        Claim(
            name="action_feature_error_decreases_with_intervention_repeats",
            passed=repeat_one["action_feature_mae"] - repeat_sixteen["action_feature_mae"] > 0.12,
            observed=round(repeat_one["action_feature_mae"] - repeat_sixteen["action_feature_mae"], 6),
            threshold="> 0.12 MAE reduction",
            detail="More sampled interventions should improve the estimated action-consequence representation.",
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


def _random_action_evidence_control() -> video.VideoRepresentationResult:
    rng = np.random.default_rng(31)
    train = video.generate_video_records("train", 96, 8, 0.015, rng)
    test = video.generate_video_records("test", 96, 8, 0.015, rng)
    train_bad = tuple(
        replace(record, action_features=tuple(float(value) for value in rng.random(len(record.action_features))))
        for record in train
    )
    test_bad = tuple(
        replace(record, action_features=tuple(float(value) for value in rng.random(len(record.action_features))))
        for record in test
    )
    return video.evaluate_action_representation(train_bad, test_bad, 2.0, 31)


def _permuted_test_action_evidence_control() -> video.VideoRepresentationResult:
    rng = np.random.default_rng(31)
    train = video.generate_video_records("train", 96, 8, 0.015, rng)
    test = video.generate_video_records("test", 96, 8, 0.015, rng)
    permutation = rng.permutation(len(test))
    test_bad = tuple(
        replace(record, action_features=test[permutation[index]].action_features)
        for index, record in enumerate(test)
    )
    return video.evaluate_action_representation(train, test_bad, 2.0, 31)


if __name__ == "__main__":
    main()
