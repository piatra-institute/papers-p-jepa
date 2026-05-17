"""Video-style representation benchmark for P-JEPA.

This is a local surrogate for the question "does P-JEPA learn a more useful
representation than a passive JEPA-like predictor?" It is not a V-JEPA
benchmark. The passive baseline learns to predict a future rendered frame from
context frames, then uses the predicted target embedding for clustering. The
P-representation learner clusters sampled intervention-outcome predictions.

The environment is designed so passive video is easy but insufficient for
control: visual styles shift between train and test, while hidden action
regimes preserve their intervention consequences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pjepa_sim.core.dishworld import ACTION_MODEL, DIRECT_ACTIONS, PROBE_LIKELIHOOD, PROBES, REGIMES
from pjepa_sim.representation.clustering import assign_nearest, kmeans, standardise
from pjepa_sim.representation.learning import best_utility_action


IMAGE_SIZE = 8
CONTEXT_FRAMES = 3
VIDEO_STYLES = ("plain", "striped")
TRAIN_STYLE = {
    "dry": "plain",
    "soapy": "plain",
    "cracked": "striped",
    "heavy": "striped",
}
TEST_STYLE = {
    "dry": "striped",
    "soapy": "plain",
    "cracked": "plain",
    "heavy": "striped",
}


@dataclass(frozen=True)
class VideoRecord:
    split: str
    context_id: int
    regime: str
    style: str
    phase: int
    context: tuple[float, ...]
    target: tuple[float, ...]
    action_features: tuple[float, ...]


@dataclass(frozen=True)
class VideoRepresentationResult:
    name: str
    success_rate: float
    unsafe_failure_rate: float
    risk_adjusted_score: float
    cluster_purity: float
    passive_prediction_mae: float | None
    action_feature_mae: float | None
    by_regime: dict[str, dict[str, float | str]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "success_rate": self.success_rate,
            "unsafe_failure_rate": self.unsafe_failure_rate,
            "risk_adjusted_score": self.risk_adjusted_score,
            "cluster_purity": self.cluster_purity,
            "passive_prediction_mae": self.passive_prediction_mae,
            "action_feature_mae": self.action_feature_mae,
            "by_regime": self.by_regime,
        }


def run_video_representation_benchmark(
    contexts_per_regime: int = 96,
    intervention_repeats: int = 8,
    pixel_noise: float = 0.015,
    seed: int = 31,
    unsafe_weight: float = 2.0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    train = generate_video_records("train", contexts_per_regime, intervention_repeats, pixel_noise, rng)
    test = generate_video_records("test", contexts_per_regime, intervention_repeats, pixel_noise, rng)
    passive_model = fit_ridge(
        np.asarray([record.context for record in train], dtype=float),
        np.asarray([record.target for record in train], dtype=float),
        l2=1e-3,
    )
    learners = {
        "prior_average": evaluate_prior(train, test, unsafe_weight),
        "jepa_passive_video": evaluate_passive_video(train, test, passive_model, unsafe_weight, seed),
        "p_action_representation": evaluate_action_representation(train, test, unsafe_weight, seed),
        "oracle_regime": evaluate_oracle(test, unsafe_weight),
    }
    return {
        "benchmark": "video_representation_surrogate",
        "description": (
            "Passive video prediction is compared with action-conditioned predicted-test "
            "representations under visual style shift. This is a local JEPA surrogate, not V-JEPA."
        ),
        "config": {
            "regimes": list(REGIMES),
            "direct_actions": list(DIRECT_ACTIONS),
            "probes": list(PROBES),
            "image_size": IMAGE_SIZE,
            "context_frames": CONTEXT_FRAMES,
            "contexts_per_regime": contexts_per_regime,
            "intervention_repeats": intervention_repeats,
            "pixel_noise": pixel_noise,
            "unsafe_weight": unsafe_weight,
            "seed": seed,
            "train_style_mapping": TRAIN_STYLE,
            "test_style_mapping": TEST_STYLE,
            "hidden_labels_used_as_features": False,
            "actual_v_jepa_or_video_foundation_model_run": False,
        },
        "learners": {name: result.as_dict() for name, result in learners.items()},
    }


def generate_video_records(
    split: str,
    contexts_per_regime: int,
    intervention_repeats: int,
    pixel_noise: float,
    rng: np.random.Generator,
) -> tuple[VideoRecord, ...]:
    style_map = TRAIN_STYLE if split == "train" else TEST_STYLE
    records: list[VideoRecord] = []
    context_id = 0
    for regime in REGIMES:
        style = style_map[regime]
        for i in range(contexts_per_regime):
            phase = i % IMAGE_SIZE
            frames = [
                render_frame(style, phase + t, pixel_noise, rng)
                for t in range(CONTEXT_FRAMES + 1)
            ]
            context = np.concatenate([frame.ravel() for frame in frames[:CONTEXT_FRAMES]])
            records.append(
                VideoRecord(
                    split=split,
                    context_id=context_id,
                    regime=regime,
                    style=style,
                    phase=phase,
                    context=tuple(float(value) for value in context),
                    target=tuple(float(value) for value in frames[-1].ravel()),
                    action_features=sample_action_features(regime, intervention_repeats, rng),
                )
            )
            context_id += 1
    return tuple(records)


def render_frame(style: str, phase: int, noise: float, rng: np.random.Generator) -> np.ndarray:
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=float)
    if style == "plain":
        image += 0.18
    else:
        for row in range(IMAGE_SIZE):
            image[row, :] = 0.10 if row % 2 == 0 else 0.30
    col = phase % IMAGE_SIZE
    row = (2 * phase + (0 if style == "plain" else 1)) % IMAGE_SIZE
    image[row, col] = 0.95
    image[(row + 1) % IMAGE_SIZE, col] = max(image[(row + 1) % IMAGE_SIZE, col], 0.60)
    if noise > 0.0:
        image += rng.normal(0.0, noise, size=image.shape)
    return np.clip(image, 0.0, 1.0)


def sample_action_features(regime: str, repeats: int, rng: np.random.Generator) -> tuple[float, ...]:
    values = []
    for action in DIRECT_ACTIONS:
        outcome = ACTION_MODEL[regime][action]
        success = rng.binomial(repeats, outcome.success) / repeats
        unsafe = rng.binomial(repeats, outcome.unsafe) / repeats
        values.extend((success, unsafe))
    for probe in PROBES:
        positive = rng.binomial(repeats, PROBE_LIKELIHOOD[probe][regime]) / repeats
        values.append(positive)
    return tuple(float(value) for value in values)


def fit_ridge(x: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=float)], axis=1)
    gram = x_aug.T @ x_aug
    gram += l2 * np.eye(gram.shape[0], dtype=float)
    return np.linalg.solve(gram, x_aug.T @ y)


def predict_ridge(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=float)], axis=1)
    return x_aug @ weights


def evaluate_prior(
    train: tuple[VideoRecord, ...],
    test: tuple[VideoRecord, ...],
    unsafe_weight: float,
) -> VideoRepresentationResult:
    train_assignments = np.zeros(len(train), dtype=int)
    test_assignments = np.zeros(len(test), dtype=int)
    sections = fit_sections(train, train_assignments)
    return evaluate_assigned_model(
        name="prior_average",
        test=test,
        assignments=test_assignments,
        sections=sections,
        cluster_purity=cluster_purity(test, test_assignments),
        unsafe_weight=unsafe_weight,
        passive_prediction_mae=None,
        action_feature_mae=None,
    )


def evaluate_passive_video(
    train: tuple[VideoRecord, ...],
    test: tuple[VideoRecord, ...],
    passive_model: np.ndarray,
    unsafe_weight: float,
    seed: int,
) -> VideoRepresentationResult:
    train_context = np.asarray([record.context for record in train], dtype=float)
    test_context = np.asarray([record.context for record in test], dtype=float)
    train_target_pred = predict_ridge(passive_model, train_context)
    test_target_pred = predict_ridge(passive_model, test_context)
    test_target = np.asarray([record.target for record in test], dtype=float)
    train_norm, mean, std = standardise(train_target_pred)
    test_norm = (test_target_pred - mean) / std
    train_assignments, centers = kmeans(train_norm, k=len(REGIMES), seed=seed)
    test_assignments = assign_nearest(test_norm, centers)
    sections = fit_sections(train, train_assignments)
    return evaluate_assigned_model(
        name="jepa_passive_video",
        test=test,
        assignments=test_assignments,
        sections=sections,
        cluster_purity=cluster_purity(test, test_assignments),
        unsafe_weight=unsafe_weight,
        passive_prediction_mae=float(np.mean(np.abs(test_target_pred - test_target))),
        action_feature_mae=None,
    )


def evaluate_action_representation(
    train: tuple[VideoRecord, ...],
    test: tuple[VideoRecord, ...],
    unsafe_weight: float,
    seed: int,
) -> VideoRepresentationResult:
    train_features = np.asarray([record.action_features for record in train], dtype=float)
    test_features = np.asarray([record.action_features for record in test], dtype=float)
    train_norm, mean, std = standardise(train_features)
    test_norm = (test_features - mean) / std
    train_assignments, centers = kmeans(train_norm, k=len(REGIMES), seed=seed)
    test_assignments = assign_nearest(test_norm, centers)
    sections = fit_sections(train, train_assignments)
    true_features = np.asarray([true_action_features(record.regime) for record in test], dtype=float)
    return evaluate_assigned_model(
        name="p_action_representation",
        test=test,
        assignments=test_assignments,
        sections=sections,
        cluster_purity=cluster_purity(test, test_assignments),
        unsafe_weight=unsafe_weight,
        passive_prediction_mae=None,
        action_feature_mae=float(np.mean(np.abs(test_features - true_features))),
    )


def evaluate_oracle(test: tuple[VideoRecord, ...], unsafe_weight: float) -> VideoRepresentationResult:
    regime_to_cluster = {regime: index for index, regime in enumerate(REGIMES)}
    assignments = np.asarray([regime_to_cluster[record.regime] for record in test], dtype=int)
    sections = {
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
        sections=sections,
        cluster_purity=1.0,
        unsafe_weight=unsafe_weight,
        passive_prediction_mae=None,
        action_feature_mae=0.0,
    )


def true_action_features(regime: str) -> tuple[float, ...]:
    values = []
    for action in DIRECT_ACTIONS:
        outcome = ACTION_MODEL[regime][action]
        values.extend((outcome.success, outcome.unsafe))
    for probe in PROBES:
        values.append(PROBE_LIKELIHOOD[probe][regime])
    return tuple(float(value) for value in values)


def fit_sections(
    records: tuple[VideoRecord, ...],
    assignments: np.ndarray,
) -> dict[int, dict[str, Any]]:
    sections: dict[int, dict[str, Any]] = {}
    for cluster in sorted(set(int(value) for value in assignments)):
        selected = [record for record, assignment in zip(records, assignments) if int(assignment) == cluster]
        sections[cluster] = {}
        outcome_type = type(ACTION_MODEL[REGIMES[0]][DIRECT_ACTIONS[0]])
        for action_index, action in enumerate(DIRECT_ACTIONS):
            success_index = 2 * action_index
            unsafe_index = success_index + 1
            successes = [record.action_features[success_index] for record in selected]
            unsafes = [record.action_features[unsafe_index] for record in selected]
            sections[cluster][action] = outcome_type(
                success=float(np.mean(successes)),
                unsafe=float(np.mean(unsafes)),
            )
    return sections


def evaluate_assigned_model(
    name: str,
    test: tuple[VideoRecord, ...],
    assignments: np.ndarray,
    sections: dict[int, dict[str, Any]],
    cluster_purity: float,
    unsafe_weight: float,
    passive_prediction_mae: float | None,
    action_feature_mae: float | None,
) -> VideoRepresentationResult:
    totals = {"success": 0.0, "unsafe": 0.0}
    by_regime: dict[str, dict[str, float | str]] = {}
    for regime in REGIMES:
        regime_records = [record for record in test if record.regime == regime]
        regime_success = 0.0
        regime_unsafe = 0.0
        action_counts: dict[str, int] = {}
        for record in regime_records:
            assignment = int(assignments[record.context_id])
            action = best_utility_action(sections[assignment], unsafe_weight)
            action_counts[action] = action_counts.get(action, 0) + 1
            outcome = ACTION_MODEL[record.regime][action]
            regime_success += outcome.success
            regime_unsafe += outcome.unsafe
        n_regime = float(len(regime_records))
        by_regime[regime] = {
            "dominant_action": max(action_counts, key=action_counts.get),
            "success_rate": regime_success / n_regime,
            "unsafe_failure_rate": regime_unsafe / n_regime,
        }
        totals["success"] += regime_success
        totals["unsafe"] += regime_unsafe

    n_total = float(len(test))
    success = totals["success"] / n_total
    unsafe = totals["unsafe"] / n_total
    return VideoRepresentationResult(
        name=name,
        success_rate=success,
        unsafe_failure_rate=unsafe,
        risk_adjusted_score=success - unsafe_weight * unsafe,
        cluster_purity=cluster_purity,
        passive_prediction_mae=passive_prediction_mae,
        action_feature_mae=action_feature_mae,
        by_regime=by_regime,
    )


def cluster_purity(records: tuple[VideoRecord, ...], assignments: np.ndarray) -> float:
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
