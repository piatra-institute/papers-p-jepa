"""Online cover-construction benchmark for action-grounded representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pjepa_sim.core.dishworld import ACTION_MODEL, DIRECT_ACTIONS, REGIMES
from pjepa_sim.representation.learning import (
    ContextRecord,
    TEST_VISUAL,
    TRAIN_VISUAL,
    VISUALS,
    action_features,
    generate_records,
)


@dataclass
class OnlineCluster:
    center: np.ndarray
    count: int = 1

    def update(self, feature: np.ndarray) -> None:
        self.count += 1
        self.center += (feature - self.center) / self.count


@dataclass(frozen=True)
class OnlineResult:
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


def run_online_cover_benchmark(
    stream_contexts_per_regime: int = 48,
    test_contexts_per_regime: int = 96,
    action_feature_noise: float = 0.035,
    threshold: float = 0.85,
    unsafe_weight: float = 2.0,
    seed: int = 23,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    stream = shuffled_stream(
        generate_records("train", stream_contexts_per_regime, action_feature_noise, rng),
        rng,
    )
    test = generate_records("test", test_contexts_per_regime, action_feature_noise, rng)

    snapshots = (16, 32, 64, 128, len(stream))
    curve = []
    final_results = None
    for n_seen in snapshots:
        prefix = stream[:n_seen]
        results = evaluate_online_learners(
            prefix,
            test,
            threshold=threshold,
            unsafe_weight=unsafe_weight,
        )
        curve.append(
            {
                "n_contexts": n_seen,
                "learners": {name: result.as_dict() for name, result in results.items()},
            }
        )
        final_results = results

    if final_results is None:
        raise RuntimeError("online benchmark produced no snapshots")

    return {
        "benchmark": "online_cover_construction",
        "description": (
            "An unlabeled stream is used to construct local action regimes online. "
            "Visual labels are shifted at test time, so useful covers must be based on "
            "action-consequence fingerprints rather than appearance."
        ),
        "config": {
            "regimes": list(REGIMES),
            "direct_actions": list(DIRECT_ACTIONS),
            "train_visual_mapping": TRAIN_VISUAL,
            "test_visual_mapping": TEST_VISUAL,
            "stream_contexts_per_regime": stream_contexts_per_regime,
            "test_contexts_per_regime": test_contexts_per_regime,
            "action_feature_noise": action_feature_noise,
            "threshold": threshold,
            "unsafe_weight": unsafe_weight,
            "seed": seed,
        },
        "learning_curve": curve,
        "learners": {name: result.as_dict() for name, result in final_results.items()},
    }


def shuffled_stream(records: tuple[ContextRecord, ...], rng: np.random.Generator) -> tuple[ContextRecord, ...]:
    indices = np.arange(len(records))
    rng.shuffle(indices)
    return tuple(records[int(index)] for index in indices)


def evaluate_online_learners(
    stream: tuple[ContextRecord, ...],
    test: tuple[ContextRecord, ...],
    threshold: float,
    unsafe_weight: float,
) -> dict[str, OnlineResult]:
    return {
        "prior_average": evaluate_prior(stream, test, unsafe_weight),
        "appearance_online": evaluate_appearance(stream, test, unsafe_weight),
        "online_action_cover": evaluate_action_cover(stream, test, threshold, unsafe_weight),
        "oracle_regime": evaluate_oracle(test, unsafe_weight),
    }


def evaluate_prior(
    stream: tuple[ContextRecord, ...],
    test: tuple[ContextRecord, ...],
    unsafe_weight: float,
) -> OnlineResult:
    centers = [np.mean(np.asarray([record.action_features for record in stream], dtype=float), axis=0)]
    assignments = np.zeros(len(test), dtype=int)
    return evaluate_centers("prior_average", test, centers, assignments, unsafe_weight)


def evaluate_appearance(
    stream: tuple[ContextRecord, ...],
    test: tuple[ContextRecord, ...],
    unsafe_weight: float,
) -> OnlineResult:
    centers: list[np.ndarray] = []
    visual_to_cluster = {}
    for visual in VISUALS:
        selected = [record.action_features for record in stream if record.visual == visual]
        if not selected:
            continue
        visual_to_cluster[visual] = len(centers)
        centers.append(np.mean(np.asarray(selected, dtype=float), axis=0))
    assignments = np.asarray(
        [visual_to_cluster.get(record.visual, 0) for record in test],
        dtype=int,
    )
    return evaluate_centers("appearance_online", test, centers, assignments, unsafe_weight)


def evaluate_action_cover(
    stream: tuple[ContextRecord, ...],
    test: tuple[ContextRecord, ...],
    threshold: float,
    unsafe_weight: float,
) -> OnlineResult:
    clusters: list[OnlineCluster] = []
    for record in stream:
        feature = np.asarray(record.action_features, dtype=float)
        if not clusters:
            clusters.append(OnlineCluster(center=feature.copy()))
            continue
        distances = [float(np.linalg.norm(feature - cluster.center)) for cluster in clusters]
        nearest = int(np.argmin(distances))
        if distances[nearest] > threshold:
            clusters.append(OnlineCluster(center=feature.copy()))
        else:
            clusters[nearest].update(feature)
    centers = [cluster.center for cluster in clusters]
    assignments = assign_nearest(np.asarray([record.action_features for record in test], dtype=float), centers)
    return evaluate_centers("online_action_cover", test, centers, assignments, unsafe_weight)


def evaluate_oracle(test: tuple[ContextRecord, ...], unsafe_weight: float) -> OnlineResult:
    centers = [
        np.asarray(action_features(regime, noise=0.0, rng=np.random.default_rng(0)), dtype=float)
        for regime in REGIMES
    ]
    regime_to_cluster = {regime: index for index, regime in enumerate(REGIMES)}
    assignments = np.asarray([regime_to_cluster[record.regime] for record in test], dtype=int)
    return evaluate_centers("oracle_regime", test, centers, assignments, unsafe_weight)


def evaluate_centers(
    name: str,
    test: tuple[ContextRecord, ...],
    centers: list[np.ndarray],
    assignments: np.ndarray,
    unsafe_weight: float,
) -> OnlineResult:
    total_success = 0.0
    total_unsafe = 0.0
    by_regime: dict[str, dict[str, float | str]] = {}
    for regime in REGIMES:
        records = [record for record in test if record.regime == regime]
        success = 0.0
        unsafe = 0.0
        action_counts: dict[str, int] = {}
        for record in records:
            cluster = int(assignments[record.context_id])
            action = best_action_from_feature(centers[cluster], unsafe_weight)
            action_counts[action] = action_counts.get(action, 0) + 1
            outcome = ACTION_MODEL[record.regime][action]
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
    return OnlineResult(
        name=name,
        success_rate=success_rate,
        unsafe_failure_rate=unsafe_rate,
        risk_adjusted_score=success_rate - unsafe_weight * unsafe_rate,
        cluster_purity=assignment_purity(test, assignments),
        n_clusters=len(centers),
        by_regime=by_regime,
    )


def best_action_from_feature(feature: np.ndarray, unsafe_weight: float) -> str:
    utilities = []
    for index, action in enumerate(DIRECT_ACTIONS):
        success = float(feature[2 * index])
        unsafe = float(feature[2 * index + 1])
        utilities.append((success - unsafe_weight * unsafe, action))
    utilities.sort(key=lambda item: (-item[0], item[1]))
    return utilities[0][1]


def assign_nearest(features: np.ndarray, centers: list[np.ndarray]) -> np.ndarray:
    matrix = np.asarray(centers, dtype=float)
    distances = np.asarray([np.sum((features - center) ** 2, axis=1) for center in matrix])
    return np.argmin(distances, axis=0)


def assignment_purity(records: tuple[ContextRecord, ...], assignments: np.ndarray) -> float:
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
