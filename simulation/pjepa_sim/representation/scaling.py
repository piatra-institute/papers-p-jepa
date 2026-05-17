"""Synthetic scaling benchmark for action-grounded representations."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
from typing import Any

import numpy as np

from pjepa_sim.representation.clustering import assign_nearest, best_kmeans, standardise


DEFAULT_REGIME_COUNTS = (4, 8, 16, 32)


@dataclass(frozen=True)
class SyntheticOutcome:
    success: float
    unsafe: float


@dataclass(frozen=True)
class ScalingRecord:
    split: str
    context_id: int
    regime: str
    visual: str
    visual_features: tuple[float, ...]
    action_features: tuple[float, ...]


@dataclass(frozen=True)
class ScalingResult:
    name: str
    success_rate: float
    unsafe_failure_rate: float
    risk_adjusted_score: float
    cluster_purity: float
    n_clusters: int
    by_regime: dict[str, dict[str, float | str]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "success_rate": self.success_rate,
            "unsafe_failure_rate": self.unsafe_failure_rate,
            "risk_adjusted_score": self.risk_adjusted_score,
            "cluster_purity": self.cluster_purity,
            "n_clusters": self.n_clusters,
            "by_regime": self.by_regime,
        }


def run_scaling_benchmark(
    regime_counts: tuple[int, ...] = DEFAULT_REGIME_COUNTS,
    contexts_per_regime: int = 72,
    n_actions: int = 8,
    n_visuals: int = 2,
    feature_noise: float = 0.012,
    unsafe_weight: float = 2.0,
    seed: int = 41,
) -> dict[str, Any]:
    cases = {}
    for offset, n_regimes in enumerate(regime_counts):
        cases[str(n_regimes)] = run_scaling_case(
            n_regimes=n_regimes,
            contexts_per_regime=contexts_per_regime,
            n_actions=n_actions,
            n_visuals=n_visuals,
            feature_noise=feature_noise,
            unsafe_weight=unsafe_weight,
            seed=seed + 997 * offset,
        )
    return {
        "benchmark": "synthetic_representation_scaling",
        "description": (
            "Synthetic hidden-regime scaling test. The number of hidden action regimes is varied while visual labels remain low-cardinality and shift between train and test. "
            "The benchmark asks whether action-consequence covers keep useful local sections as regime count grows."
        ),
        "config": {
            "regime_counts": list(regime_counts),
            "contexts_per_regime": contexts_per_regime,
            "n_actions": n_actions,
            "n_visuals": n_visuals,
            "feature_noise": feature_noise,
            "unsafe_weight": unsafe_weight,
            "seed": seed,
        },
        "cases": cases,
        "summary": summarise_cases(cases),
    }


def run_scaling_case(
    *,
    n_regimes: int,
    contexts_per_regime: int,
    n_actions: int,
    n_visuals: int,
    feature_noise: float,
    unsafe_weight: float,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    regimes = regime_names(n_regimes)
    actions = action_names(n_actions)
    model = make_action_model(regimes, actions)
    train = generate_records("train", regimes, actions, model, contexts_per_regime, n_visuals, feature_noise, rng)
    test = generate_records("test", regimes, actions, model, contexts_per_regime, n_visuals, feature_noise, rng)
    learners = {
        "prior_average": evaluate_prior(train, test, actions, model, unsafe_weight),
        "appearance_grouping": evaluate_appearance(train, test, actions, model, n_visuals, unsafe_weight),
        "action_consequence_grouping": evaluate_action_consequence(train, test, actions, model, n_regimes, unsafe_weight, seed),
        "oracle_regime": evaluate_oracle(test, actions, model, regimes, unsafe_weight),
    }
    return {
        "n_regimes": n_regimes,
        "regimes": list(regimes),
        "actions": list(actions),
        "train_visual_mapping": {regime: visual_for_regime(index, "train", n_visuals) for index, regime in enumerate(regimes)},
        "test_visual_mapping": {regime: visual_for_regime(index, "test", n_visuals) for index, regime in enumerate(regimes)},
        "learners": {name: result.as_dict() for name, result in learners.items()},
    }


def regime_names(n_regimes: int) -> tuple[str, ...]:
    return tuple(f"r{index:02d}" for index in range(n_regimes))


def action_names(n_actions: int) -> tuple[str, ...]:
    return tuple(f"a{index:02d}" for index in range(n_actions))


def make_action_model(
    regimes: tuple[str, ...],
    actions: tuple[str, ...],
) -> dict[str, dict[str, SyntheticOutcome]]:
    n_regimes = len(regimes)
    n_actions = len(actions)
    model: dict[str, dict[str, SyntheticOutcome]] = {}
    for r_index, regime in enumerate(regimes):
        theta = 2.0 * pi * r_index / n_regimes
        model[regime] = {}
        for a_index, action in enumerate(actions):
            phi = 2.0 * pi * a_index / n_actions
            alignment = (cos(theta - phi) + 1.0) / 2.0
            roughness = (sin(2.0 * theta + phi) + 1.0) / 2.0
            success = 0.35 + 0.55 * alignment
            unsafe = 0.04 + 0.42 * (1.0 - alignment) * (0.65 + 0.35 * roughness)
            model[regime][action] = SyntheticOutcome(
                success=float(np.clip(success, 0.0, 1.0)),
                unsafe=float(np.clip(unsafe, 0.0, 1.0)),
            )
    return model


def generate_records(
    split: str,
    regimes: tuple[str, ...],
    actions: tuple[str, ...],
    model: dict[str, dict[str, SyntheticOutcome]],
    contexts_per_regime: int,
    n_visuals: int,
    feature_noise: float,
    rng: np.random.Generator,
) -> tuple[ScalingRecord, ...]:
    records: list[ScalingRecord] = []
    context_id = 0
    for index, regime in enumerate(regimes):
        visual = visual_for_regime(index, split, n_visuals)
        for _ in range(contexts_per_regime):
            records.append(
                ScalingRecord(
                    split=split,
                    context_id=context_id,
                    regime=regime,
                    visual=visual,
                    visual_features=visual_features(visual, n_visuals),
                    action_features=action_features(regime, actions, model, feature_noise, rng),
                )
            )
            context_id += 1
    return tuple(records)


def visual_for_regime(regime_index: int, split: str, n_visuals: int) -> str:
    if split == "train":
        visual_index = regime_index % n_visuals
    else:
        visual_index = (regime_index // max(1, n_visuals)) % n_visuals
    return f"v{visual_index}"


def visual_features(visual: str, n_visuals: int) -> tuple[float, ...]:
    return tuple(1.0 if visual == f"v{index}" else 0.0 for index in range(n_visuals))


def action_features(
    regime: str,
    actions: tuple[str, ...],
    model: dict[str, dict[str, SyntheticOutcome]],
    feature_noise: float,
    rng: np.random.Generator,
) -> tuple[float, ...]:
    values = []
    for action in actions:
        outcome = model[regime][action]
        values.extend((outcome.success, outcome.unsafe))
    noisy = np.asarray(values, dtype=float) + rng.normal(0.0, feature_noise, size=len(values))
    return tuple(float(np.clip(value, 0.0, 1.0)) for value in noisy)


def evaluate_prior(
    train: tuple[ScalingRecord, ...],
    test: tuple[ScalingRecord, ...],
    actions: tuple[str, ...],
    model: dict[str, dict[str, SyntheticOutcome]],
    unsafe_weight: float,
) -> ScalingResult:
    train_assignments = np.zeros(len(train), dtype=int)
    test_assignments = np.zeros(len(test), dtype=int)
    sections = fit_sections(train, train_assignments, actions, model)
    return evaluate_assigned_model("prior_average", test, test_assignments, sections, actions, model, unsafe_weight)


def evaluate_appearance(
    train: tuple[ScalingRecord, ...],
    test: tuple[ScalingRecord, ...],
    actions: tuple[str, ...],
    model: dict[str, dict[str, SyntheticOutcome]],
    n_visuals: int,
    unsafe_weight: float,
) -> ScalingResult:
    visual_to_cluster = {f"v{index}": index for index in range(n_visuals)}
    train_assignments = np.asarray([visual_to_cluster[record.visual] for record in train], dtype=int)
    test_assignments = np.asarray([visual_to_cluster[record.visual] for record in test], dtype=int)
    sections = fit_sections(train, train_assignments, actions, model)
    return evaluate_assigned_model("appearance_grouping", test, test_assignments, sections, actions, model, unsafe_weight)


def evaluate_action_consequence(
    train: tuple[ScalingRecord, ...],
    test: tuple[ScalingRecord, ...],
    actions: tuple[str, ...],
    model: dict[str, dict[str, SyntheticOutcome]],
    n_regimes: int,
    unsafe_weight: float,
    seed: int,
) -> ScalingResult:
    train_features = np.asarray([record.action_features for record in train], dtype=float)
    test_features = np.asarray([record.action_features for record in test], dtype=float)
    train_norm, mean, std = standardise(train_features)
    test_norm = (test_features - mean) / std
    train_assignments, centers = best_kmeans(train_norm, k=n_regimes, seed=seed, restarts=32)
    test_assignments = assign_nearest(test_norm, centers)
    sections = fit_sections(train, train_assignments, actions, model)
    return evaluate_assigned_model("action_consequence_grouping", test, test_assignments, sections, actions, model, unsafe_weight)


def evaluate_oracle(
    test: tuple[ScalingRecord, ...],
    actions: tuple[str, ...],
    model: dict[str, dict[str, SyntheticOutcome]],
    regimes: tuple[str, ...],
    unsafe_weight: float,
) -> ScalingResult:
    regime_to_cluster = {regime: index for index, regime in enumerate(regimes)}
    assignments = np.asarray([regime_to_cluster[record.regime] for record in test], dtype=int)
    sections = {
        regime_to_cluster[regime]: {
            action: model[regime][action]
            for action in actions
        }
        for regime in regimes
    }
    return evaluate_assigned_model("oracle_regime", test, assignments, sections, actions, model, unsafe_weight)


def fit_sections(
    records: tuple[ScalingRecord, ...],
    assignments: np.ndarray,
    actions: tuple[str, ...],
    model: dict[str, dict[str, SyntheticOutcome]],
) -> dict[int, dict[str, SyntheticOutcome]]:
    sections: dict[int, dict[str, SyntheticOutcome]] = {}
    for cluster in sorted(set(int(value) for value in assignments)):
        selected = [record for record, assignment in zip(records, assignments) if int(assignment) == cluster]
        sections[cluster] = {}
        for action in actions:
            outcomes = [model[record.regime][action] for record in selected]
            sections[cluster][action] = SyntheticOutcome(
                success=float(np.mean([outcome.success for outcome in outcomes])),
                unsafe=float(np.mean([outcome.unsafe for outcome in outcomes])),
            )
    return sections


def evaluate_assigned_model(
    name: str,
    test: tuple[ScalingRecord, ...],
    assignments: np.ndarray,
    sections: dict[int, dict[str, SyntheticOutcome]],
    actions: tuple[str, ...],
    model: dict[str, dict[str, SyntheticOutcome]],
    unsafe_weight: float,
) -> ScalingResult:
    total_success = 0.0
    total_unsafe = 0.0
    by_regime: dict[str, dict[str, float | str]] = {}
    for regime in sorted({record.regime for record in test}):
        records = [record for record in test if record.regime == regime]
        success = 0.0
        unsafe = 0.0
        action_counts: dict[str, int] = {}
        for record in records:
            assignment = int(assignments[record.context_id])
            action = best_action(sections[assignment], actions, unsafe_weight)
            action_counts[action] = action_counts.get(action, 0) + 1
            outcome = model[record.regime][action]
            success += outcome.success
            unsafe += outcome.unsafe
        total_success += success
        total_unsafe += unsafe
        n_regime = float(len(records))
        by_regime[regime] = {
            "dominant_action": max(action_counts, key=action_counts.get),
            "success_rate": success / n_regime,
            "unsafe_failure_rate": unsafe / n_regime,
        }

    n_total = float(len(test))
    success_rate = total_success / n_total
    unsafe_rate = total_unsafe / n_total
    return ScalingResult(
        name=name,
        success_rate=success_rate,
        unsafe_failure_rate=unsafe_rate,
        risk_adjusted_score=success_rate - unsafe_weight * unsafe_rate,
        cluster_purity=cluster_purity(test, assignments),
        n_clusters=len(sections),
        by_regime=by_regime,
    )


def best_action(
    section: dict[str, SyntheticOutcome],
    actions: tuple[str, ...],
    unsafe_weight: float,
) -> str:
    candidates = []
    for action in actions:
        outcome = section[action]
        candidates.append((outcome.success - unsafe_weight * outcome.unsafe, action))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1]


def cluster_purity(records: tuple[ScalingRecord, ...], assignments: np.ndarray) -> float:
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


def summarise_cases(cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    learner_names = tuple(next(iter(cases.values()))["learners"].keys())
    summary = {}
    for learner in learner_names:
        scores = [case["learners"][learner]["risk_adjusted_score"] for case in cases.values()]
        purities = [case["learners"][learner]["cluster_purity"] for case in cases.values()]
        summary[learner] = {
            "min_score": float(min(scores)),
            "max_score": float(max(scores)),
            "mean_score": float(np.mean(scores)),
            "min_purity": float(min(purities)),
            "max_purity": float(max(purities)),
        }
    return summary
