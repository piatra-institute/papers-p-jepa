"""Action-grounding challenge for the practical P-JEPA use case.

The challenge bundles the repository's strongest local tests into one
reviewer-facing benchmark. It asks whether a learner can do the practical work
that motivates P-JEPA: expose passive-representation failure, learn
action-consequence representations, repair ambiguity with safe probes, glue
local action sections, and compose skills.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pjepa_sim.perception.video_representation import run_video_representation_benchmark
from pjepa_sim.representation.composition import run_skill_composition_benchmark
from pjepa_sim.representation.gluing import run_gluing_ablation_benchmark
from pjepa_sim.representation.learning import run_representation_benchmark
from pjepa_sim.representation.neural import run_neural_benchmark
from pjepa_sim.representation.neural_active import run_neural_active_probe_benchmark


EXPECTED_COMPOSITION_CHAINS = {
    "dry": "no_prep->fast_lift",
    "soapy": "wipe->two_contact_lift",
    "cracked": "cushion->slow_lift",
    "heavy": "brace->grip_hard",
}


@dataclass(frozen=True)
class ChallengeClaim:
    name: str
    passed: bool
    observed: float | str
    threshold: str
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "observed": self.observed,
            "threshold": self.threshold,
            "detail": self.detail,
        }


def run_action_grounding_challenge() -> dict[str, Any]:
    representation = run_representation_benchmark()
    neural = run_neural_benchmark()
    active = run_neural_active_probe_benchmark()
    video = run_video_representation_benchmark()
    gluing = run_gluing_ablation_benchmark()
    composition = run_skill_composition_benchmark()

    rep_action = representation["learners"]["action_consequence_grouping"]
    rep_appearance = representation["learners"]["appearance_grouping"]
    rep_prior = representation["learners"]["prior_average"]

    neural_p = neural["learners"]["neural_p_representation"]
    neural_appearance = neural["learners"]["appearance_only_encoder"]
    neural_reference = neural["learners"]["engineered_fingerprint_reference"]

    active_no_probe = active["learners"]["learned_no_probe"]
    active_probe = active["learners"]["learned_active_probe"]
    active_entropy = active["learners"]["learned_entropy_probe"]

    passive_video = video["learners"]["jepa_passive_video"]
    p_video = video["learners"]["p_action_representation"]

    identity_glue = gluing["learners"]["identity_no_glue"]
    learned_glue = gluing["learners"]["learned_restriction_glue"]

    comp_action = composition["learners"]["action_consequence_grouping"]
    comp_appearance = composition["learners"]["appearance_grouping"]
    comp_prior = composition["learners"]["prior_average"]

    actual_chains = {
        regime: comp_action["by_regime"][regime]["dominant_chain"]
        for regime in EXPECTED_COMPOSITION_CHAINS
    }

    tasks = {
        "passive_embedding_failure": {
            "question": "Do passive or appearance-based embeddings fail when action consequences matter?",
            "source_benchmarks": ["video_representation_surrogate", "action_grounded_representation"],
            "metrics": {
                "p_action_minus_passive_video_score": p_video["risk_adjusted_score"]
                - passive_video["risk_adjusted_score"],
                "passive_video_prediction_mae": passive_video["passive_prediction_mae"],
                "passive_video_action_regime_purity": passive_video["cluster_purity"],
                "p_action_video_regime_purity": p_video["cluster_purity"],
                "action_consequence_minus_appearance_score": rep_action["risk_adjusted_score"]
                - rep_appearance["risk_adjusted_score"],
                "action_consequence_minus_prior_score": rep_action["risk_adjusted_score"]
                - rep_prior["risk_adjusted_score"],
            },
        },
        "predicted_test_representation_learning": {
            "question": "Can a learned predicted-test vector support action choice without hidden labels?",
            "source_benchmarks": ["neural_intervention_encoder"],
            "metrics": {
                "neural_p_minus_appearance_score": neural_p["risk_adjusted_score"]
                - neural_appearance["risk_adjusted_score"],
                "neural_p_regime_purity": neural_p["cluster_purity"],
                "neural_p_prediction_error": neural_p["mean_prediction_error"],
                "neural_p_minus_engineered_reference_score": neural_p["risk_adjusted_score"]
                - neural_reference["risk_adjusted_score"],
                "hidden_labels_used_as_features": neural["config"]["hidden_labels_used_as_features"],
            },
        },
        "safe_probe_repair": {
            "question": "Can the learner repair an ambiguous representation before acting?",
            "source_benchmarks": ["neural_active_probe"],
            "metrics": {
                "active_minus_no_probe_score": active_probe["risk_adjusted_score"]
                - active_no_probe["risk_adjusted_score"],
                "no_probe_minus_active_unsafe": active_no_probe["unsafe_failure_rate"]
                - active_probe["unsafe_failure_rate"],
                "active_minus_entropy_score": active_probe["risk_adjusted_score"]
                - active_entropy["risk_adjusted_score"],
                "active_mean_probes": active_probe["mean_probes"],
                "hidden_labels_used_as_features": active["config"]["hidden_labels_used_as_features"],
            },
        },
        "local_interface_gluing": {
            "question": "Do learned restriction maps improve incompatible local action sections?",
            "source_benchmarks": ["restriction_map_gluing_ablation"],
            "metrics": {
                "learned_glue_minus_identity_score": learned_glue["risk_adjusted_score"]
                - identity_glue["risk_adjusted_score"],
                "learned_to_identity_residual_ratio": learned_glue["mean_overlap_residual"]
                / identity_glue["mean_overlap_residual"],
                "learned_overlap_residual": learned_glue["mean_overlap_residual"],
                "identity_overlap_residual": identity_glue["mean_overlap_residual"],
            },
        },
        "skill_composition": {
            "question": "Does the representation support simple precondition/postcondition composition?",
            "source_benchmarks": ["skill_composition"],
            "metrics": {
                "composition_minus_appearance_score": comp_action["risk_adjusted_score"]
                - comp_appearance["risk_adjusted_score"],
                "composition_minus_prior_score": comp_action["risk_adjusted_score"]
                - comp_prior["risk_adjusted_score"],
                "composition_regime_purity": comp_action["cluster_purity"],
                "expected_chains": EXPECTED_COMPOSITION_CHAINS,
                "actual_chains": actual_chains,
            },
        },
    }

    return {
        "benchmark": "action_grounding_challenge",
        "description": (
            "A practical-use harness for P-JEPA: passive embeddings are tested against "
            "action-consequence representations, learned predicted-test vectors, safe "
            "probe repair, local interface gluing, and skill composition."
        ),
        "component_benchmarks": {
            "representation": representation["benchmark"],
            "neural": neural["benchmark"],
            "active_probe": active["benchmark"],
            "video": video["benchmark"],
            "gluing": gluing["benchmark"],
            "composition": composition["benchmark"],
        },
        "tasks": tasks,
        "limitations": (
            "The challenge reuses controlled local benchmarks. It is a practical-use "
            "test for action-grounded representation under hidden regimes, not a full "
            "robotics, real-video, or foundation-model benchmark."
        ),
    }


def evaluate_action_grounding_claims(results: dict[str, Any]) -> tuple[ChallengeClaim, ...]:
    tasks = results["tasks"]
    passive = tasks["passive_embedding_failure"]["metrics"]
    predicted = tasks["predicted_test_representation_learning"]["metrics"]
    probe = tasks["safe_probe_repair"]["metrics"]
    gluing = tasks["local_interface_gluing"]["metrics"]
    composition = tasks["skill_composition"]["metrics"]
    chain_match = composition["actual_chains"] == composition["expected_chains"]
    claims = [
        ChallengeClaim(
            name="challenge_exposes_passive_embedding_failure",
            passed=passive["p_action_minus_passive_video_score"] > 0.30
            and passive["passive_video_prediction_mae"] < 0.04
            and passive["passive_video_action_regime_purity"] < 0.70,
            observed=round(passive["p_action_minus_passive_video_score"], 6),
            threshold="> 0.30 score margin with passive MAE < 0.04 and purity < 0.70",
            detail="Passive prediction can be accurate while still failing to recover action regimes.",
        ),
        ChallengeClaim(
            name="challenge_action_consequences_beat_appearance",
            passed=passive["action_consequence_minus_appearance_score"] > 0.25
            and passive["action_consequence_minus_prior_score"] > 0.10,
            observed=round(passive["action_consequence_minus_appearance_score"], 6),
            threshold="> 0.25 versus appearance and > 0.10 versus prior",
            detail="Action-consequence grouping should transfer under visual cue shift.",
        ),
        ChallengeClaim(
            name="challenge_learns_predicted_test_representation",
            passed=predicted["neural_p_minus_appearance_score"] > 0.20
            and predicted["neural_p_regime_purity"] > 0.85
            and not predicted["hidden_labels_used_as_features"],
            observed=round(predicted["neural_p_minus_appearance_score"], 6),
            threshold="> 0.20 score margin, purity > 0.85, no hidden-label features",
            detail="A small neural encoder should learn a predicted-test representation from structured interventions.",
        ),
        ChallengeClaim(
            name="challenge_safe_probes_repair_ambiguity",
            passed=probe["active_minus_no_probe_score"] > 0.18
            and probe["no_probe_minus_active_unsafe"] > 0.06
            and probe["active_mean_probes"] > 0.50
            and not probe["hidden_labels_used_as_features"],
            observed=round(probe["active_minus_no_probe_score"], 6),
            threshold="> 0.18 score margin, > 0.06 unsafe reduction, > 0.50 probes",
            detail="The learner should use safe probes to improve decisions before acting from aliased sensors.",
        ),
        ChallengeClaim(
            name="challenge_glues_incompatible_local_sections",
            passed=gluing["learned_glue_minus_identity_score"] > 0.15
            and gluing["learned_to_identity_residual_ratio"] < 0.10,
            observed=round(gluing["learned_to_identity_residual_ratio"], 6),
            threshold="< 0.10 residual ratio and > 0.15 score margin",
            detail="Restriction maps learned from overlaps should align incompatible local action sections.",
        ),
        ChallengeClaim(
            name="challenge_composes_skill_chains",
            passed=composition["composition_minus_appearance_score"] > 0.30
            and composition["composition_minus_prior_score"] > 0.10
            and composition["composition_regime_purity"] > 0.95
            and chain_match,
            observed=", ".join(f"{key}:{value}" for key, value in composition["actual_chains"].items()),
            threshold=", ".join(f"{key}:{value}" for key, value in composition["expected_chains"].items()),
            detail="The action-grounded representation should select intended precondition/postcondition chains.",
        ),
    ]
    claims.append(
        ChallengeClaim(
            name="challenge_passes_all_practical_use_steps",
            passed=all(claim.passed for claim in claims),
            observed=float(sum(1 for claim in claims if claim.passed)),
            threshold=f"= {len(claims)} component claims",
            detail="All practical-use steps should pass together in the challenge harness.",
        )
    )
    return tuple(claims)


def markdown_report(results: dict[str, Any]) -> str:
    claims = evaluate_action_grounding_claims(results)
    n_passed = sum(1 for claim in claims if claim.passed)
    lines = [
        "# Action-Grounding Challenge",
        "",
        results["description"],
        "",
        f"Passed: {n_passed} / {len(claims)}",
        "",
        "| Task | Key metric | Value |",
        "|---|---|---:|",
    ]
    task_metrics = {
        "passive_embedding_failure": (
            "p_action_minus_passive_video_score",
            results["tasks"]["passive_embedding_failure"]["metrics"]["p_action_minus_passive_video_score"],
        ),
        "predicted_test_representation_learning": (
            "neural_p_minus_appearance_score",
            results["tasks"]["predicted_test_representation_learning"]["metrics"][
                "neural_p_minus_appearance_score"
            ],
        ),
        "safe_probe_repair": (
            "active_minus_no_probe_score",
            results["tasks"]["safe_probe_repair"]["metrics"]["active_minus_no_probe_score"],
        ),
        "local_interface_gluing": (
            "learned_glue_minus_identity_score",
            results["tasks"]["local_interface_gluing"]["metrics"]["learned_glue_minus_identity_score"],
        ),
        "skill_composition": (
            "composition_minus_appearance_score",
            results["tasks"]["skill_composition"]["metrics"]["composition_minus_appearance_score"],
        ),
    }
    for task, (metric, value) in task_metrics.items():
        lines.append(f"| `{task}` | `{metric}` | {float(value):.3f} |")
    lines.extend(
        [
            "",
            "## Claim Checks",
            "",
            "| Claim | Pass | Observed | Threshold |",
            "|---|---:|---|---|",
        ]
    )
    for claim in claims:
        status = "yes" if claim.passed else "no"
        lines.append(
            f"| `{claim.name}` | {status} | {_format_markdown_value(claim.observed)} | {claim.threshold} |"
        )
    lines.extend(
        [
            "",
            "## Component Benchmarks",
            "",
        ]
    )
    for label, benchmark in results["component_benchmarks"].items():
        lines.append(f"- `{label}`: `{benchmark}`")
    lines.extend(
        [
            "",
            "Limitations: " + results["limitations"],
            "",
        ]
    )
    return "\n".join(lines)


def _format_markdown_value(value: float | str) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")
