"""Neural intervention encoder benchmark for P-representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from pjepa_sim.core.dishworld import (
    ACTION_MODEL,
    DIRECT_ACTIONS,
    PROBE_LIKELIHOOD,
    PROBE_UNSAFE,
    PROBES,
    REGIMES,
)
from pjepa_sim.representation.clustering import assign_nearest, best_kmeans, standardise
from pjepa_sim.representation.learning import TEST_VISUAL, TRAIN_VISUAL, VISUALS, visual_features


TESTS = DIRECT_ACTIONS + PROBES
SENSOR_BASE: dict[str, tuple[float, ...]] = {
    "dry": (0.10, 0.08, 0.18, 0.15),
    "soapy": (0.92, 0.10, 0.20, 0.85),
    "cracked": (0.12, 0.90, 0.23, 0.20),
    "heavy": (0.18, 0.22, 0.94, 0.25),
}


@dataclass(frozen=True)
class NeuralContext:
    split: str
    context_id: int
    regime: str
    visual: str
    visual_features: tuple[float, ...]
    sensor_features: tuple[float, ...]
    nuisance_features: tuple[float, ...]
    observation_features: tuple[float, ...]


@dataclass(frozen=True)
class InterventionRecord:
    context_id: int
    regime: str
    test: str
    target: tuple[float, float]


@dataclass(frozen=True)
class NeuralResult:
    name: str
    success_rate: float
    unsafe_failure_rate: float
    risk_adjusted_score: float
    cluster_purity: float
    mean_prediction_error: float
    by_regime: dict[str, dict[str, float | str]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "success_rate": self.success_rate,
            "unsafe_failure_rate": self.unsafe_failure_rate,
            "risk_adjusted_score": self.risk_adjusted_score,
            "cluster_purity": self.cluster_purity,
            "mean_prediction_error": self.mean_prediction_error,
            "by_regime": self.by_regime,
        }


@dataclass
class TinyMLP:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    @classmethod
    def create(cls, input_dim: int, hidden_dim: int, output_dim: int, seed: int) -> "TinyMLP":
        rng = np.random.default_rng(seed)
        w1 = rng.normal(0.0, np.sqrt(2.0 / input_dim), size=(input_dim, hidden_dim))
        b1 = np.zeros(hidden_dim)
        w2 = rng.normal(0.0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, output_dim))
        b2 = np.zeros(output_dim)
        return cls(w1=w1, b1=b1, w2=w2, b2=b2)

    def predict(self, x: np.ndarray) -> np.ndarray:
        hidden = np.tanh(x @ self.w1 + self.b1)
        logits = hidden @ self.w2 + self.b2
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 900,
        learning_rate: float = 0.025,
    ) -> None:
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        params = [self.w1, self.b1, self.w2, self.b2]
        moments = [np.zeros_like(param) for param in params]
        velocities = [np.zeros_like(param) for param in params]
        n = float(len(x))
        for step in range(1, epochs + 1):
            z1 = x @ self.w1 + self.b1
            hidden = np.tanh(z1)
            logits = hidden @ self.w2 + self.b2
            pred = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))

            dlogits = (pred - y) / n
            dw2 = hidden.T @ dlogits
            db2 = dlogits.sum(axis=0)
            dhidden = dlogits @ self.w2.T
            dz1 = dhidden * (1.0 - hidden * hidden)
            dw1 = x.T @ dz1
            db1 = dz1.sum(axis=0)
            grads = [dw1, db1, dw2, db2]

            for i, (param, grad) in enumerate(zip(params, grads)):
                moments[i] = beta1 * moments[i] + (1.0 - beta1) * grad
                velocities[i] = beta2 * velocities[i] + (1.0 - beta2) * (grad * grad)
                m_hat = moments[i] / (1.0 - beta1**step)
                v_hat = velocities[i] / (1.0 - beta2**step)
                param -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)


def run_neural_benchmark(
    contexts_per_regime: int = 128,
    intervention_repeats: int = 10,
    sensor_noise: float = 0.045,
    nuisance_dim: int = 4,
    hidden_dim: int = 32,
    unsafe_weight: float = 2.0,
    seed: int = 29,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    train_contexts = generate_contexts("train", contexts_per_regime, sensor_noise, nuisance_dim, rng)
    test_contexts = generate_contexts("test", contexts_per_regime, sensor_noise, nuisance_dim, rng)
    train_records = generate_interventions(train_contexts, intervention_repeats, rng)

    appearance_sections, appearance_assignments = fit_appearance_sections(train_contexts, train_records)
    neural_model = train_model(
        train_contexts,
        train_records,
        feature_getter=lambda context: context.sensor_features,
        hidden_dim=hidden_dim,
        seed=seed + 1,
    )
    sensor_model = train_model(
        train_contexts,
        train_records,
        feature_getter=lambda context: context.sensor_features,
        hidden_dim=hidden_dim,
        seed=seed + 2,
    )

    learners = {
        "prior_average": evaluate_prior(test_contexts, train_records, unsafe_weight),
        "appearance_only_encoder": evaluate_appearance(
            test_contexts,
            appearance_sections,
            unsafe_weight,
        ),
        "sensor_only_encoder": evaluate_direct_model(
            "sensor_only_encoder",
            test_contexts,
            sensor_model,
            lambda context: context.sensor_features,
            unsafe_weight,
        ),
        "neural_p_representation": evaluate_neural_p_representation(
            train_contexts,
            test_contexts,
            train_records,
            neural_model,
            lambda context: context.sensor_features,
            unsafe_weight,
            seed,
        ),
        "engineered_fingerprint_reference": evaluate_engineered_reference(
            train_contexts,
            test_contexts,
            train_records,
            unsafe_weight,
            seed,
        ),
        "oracle_regime": evaluate_oracle(test_contexts, unsafe_weight),
    }

    return {
        "benchmark": "neural_intervention_encoder",
        "description": (
            "A small NumPy MLP is trained from intervention records. Hidden regime labels are excluded from learner inputs. "
            "The learned predicted-test vector is built from raw physical sensor observations and test identities, then used as a P-representation for local section discovery under visual cue shift."
        ),
        "config": {
            "regimes": list(REGIMES),
            "direct_actions": list(DIRECT_ACTIONS),
            "probes": list(PROBES),
            "tests": list(TESTS),
            "train_visual_mapping": TRAIN_VISUAL,
            "test_visual_mapping": TEST_VISUAL,
            "sensor_base": SENSOR_BASE,
            "contexts_per_regime": contexts_per_regime,
            "intervention_repeats": intervention_repeats,
            "sensor_noise": sensor_noise,
            "nuisance_dim": nuisance_dim,
            "hidden_dim": hidden_dim,
            "unsafe_weight": unsafe_weight,
            "seed": seed,
            "hidden_labels_used_as_features": False,
            "neural_p_inputs": "sensor_features + test_one_hot",
            "appearance_inputs": "visual_features",
        },
        "diagnostics": {
            "train_intervention_records": len(train_records),
            "appearance_train_assignments": appearance_assignments,
        },
        "learners": {name: result.as_dict() for name, result in learners.items()},
    }


def run_neural_sample_efficiency_benchmark(
    repeat_counts: tuple[int, ...] = (1, 2, 4, 8, 10),
    contexts_per_regime: int = 96,
    sensor_noise: float = 0.045,
    nuisance_dim: int = 4,
    hidden_dim: int = 32,
    unsafe_weight: float = 2.0,
    seed: int = 29,
) -> dict[str, Any]:
    cases = {}
    for repeats in repeat_counts:
        cases[str(repeats)] = run_neural_benchmark(
            contexts_per_regime=contexts_per_regime,
            intervention_repeats=repeats,
            sensor_noise=sensor_noise,
            nuisance_dim=nuisance_dim,
            hidden_dim=hidden_dim,
            unsafe_weight=unsafe_weight,
            seed=seed,
        )
    return {
        "benchmark": "neural_intervention_sample_efficiency",
        "description": (
            "The intervention repeat count is varied while contexts and random seed are held fixed. "
            "The benchmark tests whether the learned predicted-test representation remains useful under sparse sampled intervention evidence."
        ),
        "config": {
            "repeat_counts": list(repeat_counts),
            "contexts_per_regime": contexts_per_regime,
            "sensor_noise": sensor_noise,
            "nuisance_dim": nuisance_dim,
            "hidden_dim": hidden_dim,
            "unsafe_weight": unsafe_weight,
            "seed": seed,
            "hidden_labels_used_as_features": False,
        },
        "cases": cases,
        "summary": summarize_sample_efficiency_cases(cases),
    }


def summarize_sample_efficiency_cases(cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    neural_scores = []
    prior_scores = []
    appearance_scores = []
    engineered_scores = []
    neural_purities = []
    prediction_errors = []
    for case in cases.values():
        learners = case["learners"]
        neural = learners["neural_p_representation"]
        neural_scores.append(neural["risk_adjusted_score"])
        prior_scores.append(learners["prior_average"]["risk_adjusted_score"])
        appearance_scores.append(learners["appearance_only_encoder"]["risk_adjusted_score"])
        engineered_scores.append(learners["engineered_fingerprint_reference"]["risk_adjusted_score"])
        neural_purities.append(neural["cluster_purity"])
        prediction_errors.append(neural["mean_prediction_error"])
    return {
        "min_neural_score": float(min(neural_scores)),
        "min_neural_minus_prior": float(min(score - prior for score, prior in zip(neural_scores, prior_scores))),
        "min_neural_minus_appearance": float(
            min(score - appearance for score, appearance in zip(neural_scores, appearance_scores))
        ),
        "max_engineered_gap": float(max(engineered - neural for engineered, neural in zip(engineered_scores, neural_scores))),
        "min_neural_purity": float(min(neural_purities)),
        "first_prediction_error": float(prediction_errors[0]),
        "last_prediction_error": float(prediction_errors[-1]),
        "prediction_error_reduction": float(prediction_errors[0] - prediction_errors[-1]),
    }


def generate_contexts(
    split: str,
    contexts_per_regime: int,
    sensor_noise: float,
    nuisance_dim: int,
    rng: np.random.Generator,
) -> tuple[NeuralContext, ...]:
    visual_map = TRAIN_VISUAL if split == "train" else TEST_VISUAL
    contexts: list[NeuralContext] = []
    context_id = 0
    for regime in REGIMES:
        for _ in range(contexts_per_regime):
            visual = visual_map[regime]
            visual_values = visual_features(visual)
            sensor_values = np.asarray(SENSOR_BASE[regime], dtype=float) + rng.normal(
                0.0,
                sensor_noise,
                size=len(SENSOR_BASE[regime]),
            )
            sensor_values = np.clip(sensor_values, 0.0, 1.0)
            nuisance_values = rng.uniform(0.0, 1.0, size=nuisance_dim)
            observation_values = np.concatenate([visual_values, sensor_values, nuisance_values])
            contexts.append(
                NeuralContext(
                    split=split,
                    context_id=context_id,
                    regime=regime,
                    visual=visual,
                    visual_features=tuple(float(value) for value in visual_values),
                    sensor_features=tuple(float(value) for value in sensor_values),
                    nuisance_features=tuple(float(value) for value in nuisance_values),
                    observation_features=tuple(float(value) for value in observation_values),
                )
            )
            context_id += 1
    return tuple(contexts)


def generate_interventions(
    contexts: tuple[NeuralContext, ...],
    repeats: int,
    rng: np.random.Generator,
) -> tuple[InterventionRecord, ...]:
    records: list[InterventionRecord] = []
    for context in contexts:
        for test in TESTS:
            for _ in range(repeats):
                records.append(
                    InterventionRecord(
                        context_id=context.context_id,
                        regime=context.regime,
                        test=test,
                        target=sample_intervention_target(context.regime, test, rng),
                    )
                )
    return tuple(records)


def sample_intervention_target(
    regime: str,
    test: str,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if test in DIRECT_ACTIONS:
        outcome = ACTION_MODEL[regime][test]
        return (
            float(rng.binomial(1, outcome.success)),
            float(rng.binomial(1, outcome.unsafe)),
        )
    return (
        float(rng.binomial(1, PROBE_LIKELIHOOD[test][regime])),
        float(rng.binomial(1, PROBE_UNSAFE[test])),
    )


def train_model(
    contexts: tuple[NeuralContext, ...],
    records: tuple[InterventionRecord, ...],
    feature_getter: Callable[[NeuralContext], tuple[float, ...]],
    hidden_dim: int,
    seed: int,
) -> TinyMLP:
    context_by_id = {context.context_id: context for context in contexts}
    x = []
    y = []
    for record in records:
        context = context_by_id[record.context_id]
        x.append(make_model_input(feature_getter(context), record.test))
        y.append(record.target)
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    model = TinyMLP.create(x_array.shape[1], hidden_dim, y_array.shape[1], seed)
    model.fit(x_array, y_array)
    return model


def make_model_input(features: tuple[float, ...], test: str) -> np.ndarray:
    test_one_hot = np.asarray([1.0 if test == item else 0.0 for item in TESTS], dtype=float)
    return np.concatenate([np.asarray(features, dtype=float), test_one_hot])


def predicted_test_vector(
    context: NeuralContext,
    model: TinyMLP,
    feature_getter: Callable[[NeuralContext], tuple[float, ...]],
) -> tuple[float, ...]:
    values = []
    for test in TESTS:
        pred = model.predict(make_model_input(feature_getter(context), test)[None, :])[0]
        if test in DIRECT_ACTIONS:
            values.extend((float(pred[0]), float(pred[1])))
        else:
            values.append(float(pred[0]))
    return tuple(values)


def fit_appearance_sections(
    train_contexts: tuple[NeuralContext, ...],
    train_records: tuple[InterventionRecord, ...],
) -> tuple[dict[int, dict[str, tuple[float, float]]], dict[str, int]]:
    visual_to_cluster = {visual: index for index, visual in enumerate(VISUALS)}
    assignments = {
        context.context_id: visual_to_cluster[context.visual]
        for context in train_contexts
    }
    return fit_sections_from_records(train_records, assignments), visual_to_cluster


def evaluate_prior(
    test_contexts: tuple[NeuralContext, ...],
    train_records: tuple[InterventionRecord, ...],
    unsafe_weight: float,
) -> NeuralResult:
    assignments = {context.context_id: 0 for context in test_contexts}
    sections = fit_sections_from_records(train_records, {record.context_id: 0 for record in train_records})
    return evaluate_assigned_sections(
        "prior_average",
        test_contexts,
        assignments,
        sections,
        unsafe_weight,
        cluster_purity=cluster_purity(test_contexts, assignments),
        mean_prediction_error=0.0,
    )


def evaluate_appearance(
    test_contexts: tuple[NeuralContext, ...],
    sections: dict[int, dict[str, tuple[float, float]]],
    unsafe_weight: float,
) -> NeuralResult:
    visual_to_cluster = {visual: index for index, visual in enumerate(VISUALS)}
    assignments = {
        context.context_id: visual_to_cluster[context.visual]
        for context in test_contexts
    }
    return evaluate_assigned_sections(
        "appearance_only_encoder",
        test_contexts,
        assignments,
        sections,
        unsafe_weight,
        cluster_purity=cluster_purity(test_contexts, assignments),
        mean_prediction_error=0.0,
    )


def evaluate_direct_model(
    name: str,
    test_contexts: tuple[NeuralContext, ...],
    model: TinyMLP,
    feature_getter: Callable[[NeuralContext], tuple[float, ...]],
    unsafe_weight: float,
) -> NeuralResult:
    total_success = 0.0
    total_unsafe = 0.0
    total_error = 0.0
    by_regime: dict[str, dict[str, float | str]] = {}
    for regime in REGIMES:
        selected = [context for context in test_contexts if context.regime == regime]
        success = 0.0
        unsafe = 0.0
        error = 0.0
        action_counts: dict[str, int] = {}
        for context in selected:
            predictions = {
                action: model.predict(make_model_input(feature_getter(context), action)[None, :])[0]
                for action in DIRECT_ACTIONS
            }
            action = best_action_from_predictions(predictions, unsafe_weight)
            action_counts[action] = action_counts.get(action, 0) + 1
            outcome = ACTION_MODEL[context.regime][action]
            success += outcome.success
            unsafe += outcome.unsafe
            error += direct_prediction_error(context.regime, predictions)
        total_success += success
        total_unsafe += unsafe
        total_error += error
        n_regime = float(len(selected))
        by_regime[regime] = {
            "dominant_action": max(action_counts, key=action_counts.get),
            "success_rate": success / n_regime,
            "unsafe_failure_rate": unsafe / n_regime,
            "mean_prediction_error": error / n_regime,
        }

    n_total = float(len(test_contexts))
    success_rate = total_success / n_total
    unsafe_rate = total_unsafe / n_total
    prediction_vectors = np.asarray(
        [predicted_test_vector(context, model, feature_getter) for context in test_contexts],
        dtype=float,
    )
    prediction_norm, _, _ = standardise(prediction_vectors)
    assignments, _ = best_kmeans(prediction_norm, k=len(REGIMES), seed=13)
    assignment_map = {context.context_id: int(assignments[index]) for index, context in enumerate(test_contexts)}
    return NeuralResult(
        name=name,
        success_rate=success_rate,
        unsafe_failure_rate=unsafe_rate,
        risk_adjusted_score=success_rate - unsafe_weight * unsafe_rate,
        cluster_purity=cluster_purity(test_contexts, assignment_map),
        mean_prediction_error=total_error / n_total,
        by_regime=by_regime,
    )


def evaluate_neural_p_representation(
    train_contexts: tuple[NeuralContext, ...],
    test_contexts: tuple[NeuralContext, ...],
    train_records: tuple[InterventionRecord, ...],
    model: TinyMLP,
    feature_getter: Callable[[NeuralContext], tuple[float, ...]],
    unsafe_weight: float,
    seed: int,
) -> NeuralResult:
    train_vectors = np.asarray(
        [predicted_test_vector(context, model, feature_getter) for context in train_contexts],
        dtype=float,
    )
    test_vectors = np.asarray(
        [predicted_test_vector(context, model, feature_getter) for context in test_contexts],
        dtype=float,
    )
    train_norm, mean, std = standardise(train_vectors)
    test_norm = (test_vectors - mean) / std
    train_assignments, centers = best_kmeans(train_norm, k=len(REGIMES), seed=seed, restarts=32)
    test_assignments = assign_nearest(test_norm, centers)
    train_assignment_map = {
        context.context_id: int(train_assignments[index])
        for index, context in enumerate(train_contexts)
    }
    test_assignment_map = {
        context.context_id: int(test_assignments[index])
        for index, context in enumerate(test_contexts)
    }
    sections = fit_sections_from_records(train_records, train_assignment_map)
    return evaluate_assigned_sections(
        "neural_p_representation",
        test_contexts,
        test_assignment_map,
        sections,
        unsafe_weight,
        cluster_purity=cluster_purity(test_contexts, test_assignment_map),
        mean_prediction_error=mean_vector_prediction_error(test_contexts, test_vectors),
    )


def evaluate_engineered_reference(
    train_contexts: tuple[NeuralContext, ...],
    test_contexts: tuple[NeuralContext, ...],
    train_records: tuple[InterventionRecord, ...],
    unsafe_weight: float,
    seed: int,
) -> NeuralResult:
    train_vectors = np.asarray([engineered_vector(context.regime) for context in train_contexts], dtype=float)
    test_vectors = np.asarray([engineered_vector(context.regime) for context in test_contexts], dtype=float)
    train_norm, mean, std = standardise(train_vectors)
    test_norm = (test_vectors - mean) / std
    train_assignments, centers = best_kmeans(train_norm, k=len(REGIMES), seed=seed + 7, restarts=32)
    test_assignments = assign_nearest(test_norm, centers)
    train_assignment_map = {
        context.context_id: int(train_assignments[index])
        for index, context in enumerate(train_contexts)
    }
    test_assignment_map = {
        context.context_id: int(test_assignments[index])
        for index, context in enumerate(test_contexts)
    }
    sections = fit_sections_from_records(train_records, train_assignment_map)
    return evaluate_assigned_sections(
        "engineered_fingerprint_reference",
        test_contexts,
        test_assignment_map,
        sections,
        unsafe_weight,
        cluster_purity=cluster_purity(test_contexts, test_assignment_map),
        mean_prediction_error=0.0,
    )


def evaluate_oracle(
    test_contexts: tuple[NeuralContext, ...],
    unsafe_weight: float,
) -> NeuralResult:
    regime_to_cluster = {regime: index for index, regime in enumerate(REGIMES)}
    assignments = {context.context_id: regime_to_cluster[context.regime] for context in test_contexts}
    sections = {
        regime_to_cluster[regime]: {
            action: (
                ACTION_MODEL[regime][action].success,
                ACTION_MODEL[regime][action].unsafe,
            )
            for action in DIRECT_ACTIONS
        }
        for regime in REGIMES
    }
    return evaluate_assigned_sections(
        "oracle_regime",
        test_contexts,
        assignments,
        sections,
        unsafe_weight,
        cluster_purity=1.0,
        mean_prediction_error=0.0,
    )


def fit_sections_from_records(
    records: tuple[InterventionRecord, ...],
    context_assignments: dict[int, int],
) -> dict[int, dict[str, tuple[float, float]]]:
    sections: dict[int, dict[str, tuple[float, float]]] = {}
    global_section = empirical_section(records)
    for cluster in sorted(set(context_assignments.values())):
        selected = [
            record
            for record in records
            if context_assignments.get(record.context_id) == cluster
        ]
        sections[cluster] = empirical_section(selected, fallback=global_section)
    return sections


def empirical_section(
    records: list[InterventionRecord] | tuple[InterventionRecord, ...],
    fallback: dict[str, tuple[float, float]] | None = None,
) -> dict[str, tuple[float, float]]:
    section = {}
    for action in DIRECT_ACTIONS:
        selected = [record for record in records if record.test == action]
        if selected:
            section[action] = (
                float(np.mean([record.target[0] for record in selected])),
                float(np.mean([record.target[1] for record in selected])),
            )
        elif fallback is not None:
            section[action] = fallback[action]
        else:
            section[action] = (0.5, 0.5)
    return section


def evaluate_assigned_sections(
    name: str,
    test_contexts: tuple[NeuralContext, ...],
    assignments: dict[int, int],
    sections: dict[int, dict[str, tuple[float, float]]],
    unsafe_weight: float,
    cluster_purity: float,
    mean_prediction_error: float,
) -> NeuralResult:
    total_success = 0.0
    total_unsafe = 0.0
    by_regime: dict[str, dict[str, float | str]] = {}
    for regime in REGIMES:
        selected = [context for context in test_contexts if context.regime == regime]
        success = 0.0
        unsafe = 0.0
        action_counts: dict[str, int] = {}
        for context in selected:
            cluster = assignments[context.context_id]
            action = best_action_from_section(sections[cluster], unsafe_weight)
            action_counts[action] = action_counts.get(action, 0) + 1
            outcome = ACTION_MODEL[context.regime][action]
            success += outcome.success
            unsafe += outcome.unsafe
        total_success += success
        total_unsafe += unsafe
        n_regime = float(len(selected))
        by_regime[regime] = {
            "dominant_action": max(action_counts, key=action_counts.get),
            "success_rate": success / n_regime,
            "unsafe_failure_rate": unsafe / n_regime,
        }

    n_total = float(len(test_contexts))
    success_rate = total_success / n_total
    unsafe_rate = total_unsafe / n_total
    return NeuralResult(
        name=name,
        success_rate=success_rate,
        unsafe_failure_rate=unsafe_rate,
        risk_adjusted_score=success_rate - unsafe_weight * unsafe_rate,
        cluster_purity=cluster_purity,
        mean_prediction_error=mean_prediction_error,
        by_regime=by_regime,
    )


def best_action_from_section(
    section: dict[str, tuple[float, float]],
    unsafe_weight: float,
) -> str:
    utilities = {
        action: section[action][0] - unsafe_weight * section[action][1]
        for action in DIRECT_ACTIONS
    }
    return max(utilities, key=utilities.get)


def best_action_from_predictions(
    predictions: dict[str, np.ndarray],
    unsafe_weight: float,
) -> str:
    utilities = {
        action: float(prediction[0] - unsafe_weight * prediction[1])
        for action, prediction in predictions.items()
    }
    return max(utilities, key=utilities.get)


def direct_prediction_error(
    regime: str,
    predictions: dict[str, np.ndarray],
) -> float:
    errors = []
    for action, prediction in predictions.items():
        outcome = ACTION_MODEL[regime][action]
        target = np.asarray([outcome.success, outcome.unsafe], dtype=float)
        errors.append(float(np.mean(np.abs(prediction - target))))
    return float(np.mean(errors))


def mean_vector_prediction_error(
    contexts: tuple[NeuralContext, ...],
    vectors: np.ndarray,
) -> float:
    targets = np.asarray([engineered_vector(context.regime) for context in contexts], dtype=float)
    return float(np.mean(np.abs(vectors - targets)))


def engineered_vector(regime: str) -> tuple[float, ...]:
    values = []
    for action in DIRECT_ACTIONS:
        outcome = ACTION_MODEL[regime][action]
        values.extend((outcome.success, outcome.unsafe))
    for probe in PROBES:
        values.append(PROBE_LIKELIHOOD[probe][regime])
    return tuple(float(value) for value in values)


def cluster_purity(
    contexts: tuple[NeuralContext, ...],
    assignments: dict[int, int],
) -> float:
    total = 0
    correct = 0
    for cluster in sorted(set(assignments.values())):
        labels = [
            context.regime
            for context in contexts
            if assignments[context.context_id] == cluster
        ]
        if not labels:
            continue
        counts = {label: labels.count(label) for label in set(labels)}
        correct += max(counts.values())
        total += len(labels)
    return float(correct / total) if total else 0.0
