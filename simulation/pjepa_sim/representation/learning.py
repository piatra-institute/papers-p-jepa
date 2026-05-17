"""Action-grounded representation benchmark.

The benchmark tests the representation-learning claim that visually grouped
contexts are not enough when visually similar situations have different action
consequences. Learners receive context records with visual features and
action/probe fingerprints. Hidden regime labels are used only for evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pjepa_sim.core.dishworld import ACTION_MODEL, DIRECT_ACTIONS, PROBE_LIKELIHOOD, PROBES, REGIMES
from pjepa_sim.representation.clustering import assign_nearest, kmeans, standardise


TRAIN_VISUAL = {
    "dry": "white",
    "soapy": "white",
    "cracked": "blue",
    "heavy": "blue",
}
TEST_VISUAL = {
    "dry": "blue",
    "soapy": "white",
    "cracked": "white",
    "heavy": "blue",
}
VISUALS = ("white", "blue")


@dataclass(frozen=True)
class ContextRecord:
    split: str
    context_id: int
    regime: str
    visual: str
    visual_features: tuple[float, ...]
    action_features: tuple[float, ...]


@dataclass(frozen=True)
class RepresentationResult:
    name: str
    success_rate: float
    unsafe_failure_rate: float
    risk_adjusted_score: float
    cluster_purity: float
    mean_feature_distance: float
    by_regime: dict[str, dict[str, float | str]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "success_rate": self.success_rate,
            "unsafe_failure_rate": self.unsafe_failure_rate,
            "risk_adjusted_score": self.risk_adjusted_score,
            "cluster_purity": self.cluster_purity,
            "mean_feature_distance": self.mean_feature_distance,
            "by_regime": self.by_regime,
        }


def run_representation_benchmark(
    contexts_per_regime: int = 96,
    action_feature_noise: float = 0.035,
    seed: int = 7,
    unsafe_weight: float = 2.0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    train = generate_records("train", contexts_per_regime, action_feature_noise, rng)
    test = generate_records("test", contexts_per_regime, action_feature_noise, rng)
    learners = {
        "prior_average": evaluate_prior(train, test, unsafe_weight),
        "appearance_grouping": evaluate_appearance(train, test, unsafe_weight),
        "action_consequence_grouping": evaluate_action_consequence(train, test, unsafe_weight, seed),
        "oracle_regime": evaluate_oracle(test, unsafe_weight),
    }
    return {
        "benchmark": "action_grounded_representation",
        "description": (
            "Visual cues shift between train and test. Action-consequence fingerprints "
            "remain stable and should support downstream action choice."
        ),
        "config": {
            "regimes": list(REGIMES),
            "direct_actions": list(DIRECT_ACTIONS),
            "probes": list(PROBES),
            "train_visual_mapping": TRAIN_VISUAL,
            "test_visual_mapping": TEST_VISUAL,
            "contexts_per_regime": contexts_per_regime,
            "action_feature_noise": action_feature_noise,
            "unsafe_weight": unsafe_weight,
            "seed": seed,
        },
        "learners": {name: result.as_dict() for name, result in learners.items()},
    }


def generate_records(
    split: str,
    contexts_per_regime: int,
    action_feature_noise: float,
    rng: np.random.Generator,
) -> tuple[ContextRecord, ...]:
    records: list[ContextRecord] = []
    visual_map = TRAIN_VISUAL if split == "train" else TEST_VISUAL
    context_id = 0
    for regime in REGIMES:
        for _ in range(contexts_per_regime):
            visual = visual_map[regime]
            records.append(
                ContextRecord(
                    split=split,
                    context_id=context_id,
                    regime=regime,
                    visual=visual,
                    visual_features=visual_features(visual),
                    action_features=action_features(regime, action_feature_noise, rng),
                )
            )
            context_id += 1
    return tuple(records)


def visual_features(visual: str) -> tuple[float, ...]:
    return tuple(1.0 if visual == item else 0.0 for item in VISUALS)


def action_features(regime: str, noise: float, rng: np.random.Generator) -> tuple[float, ...]:
    values = []
    for action in DIRECT_ACTIONS:
        outcome = ACTION_MODEL[regime][action]
        values.extend((outcome.success, outcome.unsafe))
    for probe in PROBES:
        values.append(PROBE_LIKELIHOOD[probe][regime])
    noisy = np.asarray(values, dtype=float) + rng.normal(0.0, noise, size=len(values))
    return tuple(float(np.clip(value, 0.0, 1.0)) for value in noisy)


def evaluate_prior(
    train: tuple[ContextRecord, ...],
    test: tuple[ContextRecord, ...],
    unsafe_weight: float,
) -> RepresentationResult:
    local_sections = fit_sections(train, np.zeros(len(train), dtype=int))
    assignments = np.zeros(len(test), dtype=int)
    return evaluate_assigned_model(
        name="prior_average",
        test=test,
        assignments=assignments,
        local_sections=local_sections,
        cluster_purity=cluster_purity(test, assignments),
        mean_feature_distance=0.0,
        unsafe_weight=unsafe_weight,
    )


def evaluate_appearance(
    train: tuple[ContextRecord, ...],
    test: tuple[ContextRecord, ...],
    unsafe_weight: float,
) -> RepresentationResult:
    visual_to_cluster = {visual: index for index, visual in enumerate(VISUALS)}
    train_assignments = np.asarray([visual_to_cluster[record.visual] for record in train], dtype=int)
    test_assignments = np.asarray([visual_to_cluster[record.visual] for record in test], dtype=int)
    local_sections = fit_sections(train, train_assignments)
    return evaluate_assigned_model(
        name="appearance_grouping",
        test=test,
        assignments=test_assignments,
        local_sections=local_sections,
        cluster_purity=cluster_purity(test, test_assignments),
        mean_feature_distance=mean_intra_cluster_distance(test, test_assignments, feature="visual"),
        unsafe_weight=unsafe_weight,
    )


def evaluate_action_consequence(
    train: tuple[ContextRecord, ...],
    test: tuple[ContextRecord, ...],
    unsafe_weight: float,
    seed: int,
) -> RepresentationResult:
    train_features = np.asarray([record.action_features for record in train], dtype=float)
    test_features = np.asarray([record.action_features for record in test], dtype=float)
    train_norm, mean, std = standardise(train_features)
    test_norm = (test_features - mean) / std
    train_assignments, centers = kmeans(train_norm, k=len(REGIMES), seed=seed)
    test_assignments = assign_nearest(test_norm, centers)
    local_sections = fit_sections(train, train_assignments)
    return evaluate_assigned_model(
        name="action_consequence_grouping",
        test=test,
        assignments=test_assignments,
        local_sections=local_sections,
        cluster_purity=cluster_purity(test, test_assignments),
        mean_feature_distance=mean_intra_cluster_distance(test, test_assignments, feature="action"),
        unsafe_weight=unsafe_weight,
    )


def evaluate_oracle(test: tuple[ContextRecord, ...], unsafe_weight: float) -> RepresentationResult:
    regime_to_cluster = {regime: index for index, regime in enumerate(REGIMES)}
    assignments = np.asarray([regime_to_cluster[record.regime] for record in test], dtype=int)
    local_sections = {
        regime_to_cluster[regime]: {
            action: ACTION_MODEL[regime][action]
            for action in DIRECT_ACTIONS
        }
        for regime in REGIMES
    }
    return evaluate_assigned_model(
        name="oracle_regime",
        test=test,
        assignments=assignments,
        local_sections=local_sections,
        cluster_purity=1.0,
        mean_feature_distance=0.0,
        unsafe_weight=unsafe_weight,
    )


def fit_sections(
    records: tuple[ContextRecord, ...],
    assignments: np.ndarray,
) -> dict[int, dict[str, Any]]:
    sections: dict[int, dict[str, Any]] = {}
    for cluster in sorted(set(int(value) for value in assignments)):
        selected = [record for record, assignment in zip(records, assignments) if int(assignment) == cluster]
        sections[cluster] = {}
        for action in DIRECT_ACTIONS:
            sections[cluster][action] = mean_outcome(selected, action)
    return sections


def mean_outcome(records: list[ContextRecord], action: str) -> Any:
    successes = [ACTION_MODEL[record.regime][action].success for record in records]
    unsafes = [ACTION_MODEL[record.regime][action].unsafe for record in records]
    return type(ACTION_MODEL[records[0].regime][action])(
        success=float(np.mean(successes)),
        unsafe=float(np.mean(unsafes)),
    )


def evaluate_assigned_model(
    name: str,
    test: tuple[ContextRecord, ...],
    assignments: np.ndarray,
    local_sections: dict[int, dict[str, Any]],
    cluster_purity: float,
    mean_feature_distance: float,
    unsafe_weight: float,
) -> RepresentationResult:
    totals = {"success": 0.0, "unsafe": 0.0}
    by_regime: dict[str, dict[str, float | str]] = {}
    for regime in REGIMES:
        regime_records = [record for record in test if record.regime == regime]
        regime_success = 0.0
        regime_unsafe = 0.0
        action_counts: dict[str, int] = {}
        for record in regime_records:
            assignment = int(assignments[record.context_id])
            action = best_utility_action(local_sections[assignment], unsafe_weight)
            action_counts[action] = action_counts.get(action, 0) + 1
            outcome = ACTION_MODEL[record.regime][action]
            regime_success += outcome.success
            regime_unsafe += outcome.unsafe
        n = float(len(regime_records))
        by_regime[regime] = {
            "dominant_action": max(action_counts, key=action_counts.get),
            "success_rate": regime_success / n,
            "unsafe_failure_rate": regime_unsafe / n,
        }
        totals["success"] += regime_success
        totals["unsafe"] += regime_unsafe

    n_total = float(len(test))
    success = totals["success"] / n_total
    unsafe = totals["unsafe"] / n_total
    return RepresentationResult(
        name=name,
        success_rate=success,
        unsafe_failure_rate=unsafe,
        risk_adjusted_score=success - unsafe_weight * unsafe,
        cluster_purity=cluster_purity,
        mean_feature_distance=mean_feature_distance,
        by_regime=by_regime,
    )


def best_utility_action(section: dict[str, Any], unsafe_weight: float) -> str:
    utilities = {
        action: outcome.success - unsafe_weight * outcome.unsafe
        for action, outcome in section.items()
    }
    return max(utilities, key=utilities.get)


def cluster_purity(records: tuple[ContextRecord, ...], assignments: np.ndarray) -> float:
    total = 0
    correct = 0
    for cluster in sorted(set(int(value) for value in assignments)):
        labels = [
            record.regime
            for record, assignment in zip(records, assignments)
            if int(assignment) == cluster
        ]
        if not labels:
            continue
        counts = {label: labels.count(label) for label in set(labels)}
        correct += max(counts.values())
        total += len(labels)
    return float(correct / total) if total else 0.0


def mean_intra_cluster_distance(
    records: tuple[ContextRecord, ...],
    assignments: np.ndarray,
    feature: str,
) -> float:
    distances = []
    for cluster in sorted(set(int(value) for value in assignments)):
        features = [
            np.asarray(record.visual_features if feature == "visual" else record.action_features)
            for record, assignment in zip(records, assignments)
            if int(assignment) == cluster
        ]
        if len(features) <= 1:
            continue
        matrix = np.asarray(features, dtype=float)
        center = matrix.mean(axis=0)
        distances.extend(np.linalg.norm(matrix - center, axis=1))
    return float(np.mean(distances)) if distances else 0.0

