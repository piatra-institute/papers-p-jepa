"""Learned hidden-regime model for the Meta-World adapter.

This module replaces the hand-specified probe and local-section model used by
the external adapter with estimates fit from sampled wrapper experience. The
true simulator config is still used to generate hidden dynamics during
evaluation; the agent's posterior updates and obstruction use the learned
config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pjepa_sim.external.metaworld_hidden_regime import (
    PROBES,
    REGIMES,
    HiddenRegime,
    HiddenRegimeMetaWorldAdapter,
    MetaWorldHiddenRegimeConfig,
    ProbeSpec,
    make_metaworld_env,
    run_strategy_benchmark,
)


@dataclass(frozen=True)
class ProbeSample:
    regime: str
    probe: str
    positive: bool
    unsafe: bool


@dataclass(frozen=True)
class ActionSample:
    regime: str
    raw_action: tuple[float, ...]
    transformed_action: tuple[float, ...]
    unsafe: bool


@dataclass(frozen=True)
class LearnedDataset:
    probe_samples: tuple[ProbeSample, ...]
    action_samples: tuple[ActionSample, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_probe_samples": len(self.probe_samples),
            "n_action_samples": len(self.action_samples),
            "probe_samples_per_regime_probe": _count_probe_samples(self.probe_samples),
            "action_samples_per_regime": _count_action_samples(self.action_samples),
        }


@dataclass(frozen=True)
class FingerprintSample:
    true_regime: str
    probe_rates: tuple[float, ...]
    action_scale: float
    action_noise: float
    unsafe_threshold: float

    def features(self) -> np.ndarray:
        return np.asarray(
            (
                *self.probe_rates,
                self.action_scale,
                self.action_noise,
                self.unsafe_threshold,
            ),
            dtype=float,
        )


@dataclass(frozen=True)
class UnsupervisedDataset:
    fingerprints: tuple[FingerprintSample, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_fingerprints": len(self.fingerprints),
            "fingerprints_per_true_regime": _count_fingerprints(self.fingerprints),
        }


@dataclass(frozen=True)
class RawRecord:
    context_id: int
    event_type: str
    probe: str | None
    probe_positive: bool | None
    raw_action: tuple[float, ...]
    transformed_action: tuple[float, ...]
    unsafe: bool


@dataclass(frozen=True)
class RawRecordDataset:
    records: tuple[RawRecord, ...]
    context_labels: dict[int, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_records": len(self.records),
            "n_contexts": len({record.context_id for record in self.records}),
            "records_per_event_type": _count_raw_events(self.records),
            "contexts_per_true_regime": _count_raw_contexts(self.context_labels),
        }


def collect_learning_data(
    config: MetaWorldHiddenRegimeConfig,
    probe_samples_per_regime_probe: int,
    action_samples_per_regime: int,
    seed: int,
) -> LearnedDataset:
    """Collect labelled wrapper experience for fitting local sections.

    The labels are the hidden regimes supplied by the simulator. This is the
    first learned step, not unsupervised regime discovery.
    """
    env = make_metaworld_env(config)
    rng = np.random.default_rng(seed)
    adapter = HiddenRegimeMetaWorldAdapter(env, config=config, rng=rng)
    probe_samples: list[ProbeSample] = []
    action_samples: list[ActionSample] = []
    try:
        for regime in REGIMES:
            for probe in PROBES:
                true_probe = config.probes[probe]
                for _ in range(probe_samples_per_regime_probe):
                    positive = bool(rng.random() < true_probe.likelihood[regime])
                    unsafe = bool(rng.random() < true_probe.unsafe_probability)
                    probe_samples.append(
                        ProbeSample(
                            regime=regime,
                            probe=probe,
                            positive=positive,
                            unsafe=unsafe,
                        )
                    )

            adapter.hidden_regime = regime
            for _ in range(action_samples_per_regime):
                raw = np.asarray(env.action_space.sample(), dtype=float)
                transformed = adapter.transform_action(raw)
                action_samples.append(
                    ActionSample(
                        regime=regime,
                        raw_action=tuple(float(x) for x in raw),
                        transformed_action=tuple(float(x) for x in transformed),
                        unsafe=adapter.is_unsafe_action(transformed),
                    )
                )
    finally:
        env.close()

    return LearnedDataset(
        probe_samples=tuple(probe_samples),
        action_samples=tuple(action_samples),
    )


def collect_unlabeled_fingerprints(
    config: MetaWorldHiddenRegimeConfig,
    contexts_per_regime: int,
    probe_trials_per_probe: int,
    action_trials_per_context: int,
    seed: int,
) -> UnsupervisedDataset:
    """Collect episode-level fingerprints without passing labels to the learner."""
    env = make_metaworld_env(config)
    rng = np.random.default_rng(seed)
    adapter = HiddenRegimeMetaWorldAdapter(env, config=config, rng=rng)
    fingerprints: list[FingerprintSample] = []
    try:
        for regime in REGIMES:
            adapter.hidden_regime = regime
            for _ in range(contexts_per_regime):
                probe_rates = []
                for probe in PROBES:
                    true_probe = config.probes[probe]
                    positives = sum(
                        1
                        for _ in range(probe_trials_per_probe)
                        if rng.random() < true_probe.likelihood[regime]
                    )
                    probe_rates.append(positives / probe_trials_per_probe)

                raw_actions = []
                transformed_actions = []
                unsafe = []
                for _ in range(action_trials_per_context):
                    raw = np.asarray(env.action_space.sample(), dtype=float)
                    transformed = adapter.transform_action(raw)
                    raw_actions.append(raw)
                    transformed_actions.append(transformed)
                    unsafe.append(adapter.is_unsafe_action(transformed))

                raw_array = np.asarray(raw_actions, dtype=float)
                transformed_array = np.asarray(transformed_actions, dtype=float)
                unsafe_array = np.asarray(unsafe, dtype=bool)
                scale = estimate_action_scale(raw_array, transformed_array)
                noise = estimate_action_noise(raw_array, transformed_array, scale)
                threshold = estimate_unsafe_threshold(transformed_array, unsafe_array)
                fingerprints.append(
                    FingerprintSample(
                        true_regime=regime,
                        probe_rates=tuple(float(x) for x in probe_rates),
                        action_scale=scale,
                        action_noise=noise,
                        unsafe_threshold=threshold,
                    )
                )
    finally:
        env.close()

    return UnsupervisedDataset(fingerprints=tuple(fingerprints))


def collect_unlabeled_stream_fingerprints(
    config: MetaWorldHiddenRegimeConfig,
    total_contexts: int,
    probe_trials_per_probe: int,
    action_trials_per_context: int,
    seed: int,
) -> UnsupervisedDataset:
    """Collect unlabeled fingerprints from a prior-sampled stream."""
    env = make_metaworld_env(config)
    rng = np.random.default_rng(seed)
    adapter = HiddenRegimeMetaWorldAdapter(env, config=config, rng=rng)
    fingerprints: list[FingerprintSample] = []
    try:
        for _ in range(total_contexts):
            regime = str(rng.choice(REGIMES, p=config.prior_array()))
            adapter.hidden_regime = regime
            fingerprints.append(
                sample_fingerprint(
                    config=config,
                    adapter=adapter,
                    rng=rng,
                    regime=regime,
                    probe_trials_per_probe=probe_trials_per_probe,
                    action_trials_per_context=action_trials_per_context,
                )
            )
    finally:
        env.close()

    return UnsupervisedDataset(fingerprints=tuple(fingerprints))


def collect_unlabeled_raw_stream_records(
    config: MetaWorldHiddenRegimeConfig,
    total_contexts: int,
    probe_trials_per_probe: int,
    action_trials_per_context: int,
    seed: int,
) -> RawRecordDataset:
    """Collect unlabeled event records from a prior-sampled context stream."""
    env = make_metaworld_env(config)
    rng = np.random.default_rng(seed)
    adapter = HiddenRegimeMetaWorldAdapter(env, config=config, rng=rng)
    records: list[RawRecord] = []
    context_labels: dict[int, str] = {}
    try:
        for context_id in range(total_contexts):
            regime = str(rng.choice(REGIMES, p=config.prior_array()))
            adapter.hidden_regime = regime
            context_labels[context_id] = regime

            for probe in PROBES:
                true_probe = config.probes[probe]
                for _ in range(probe_trials_per_probe):
                    records.append(
                        RawRecord(
                            context_id=context_id,
                            event_type="probe",
                            probe=probe,
                            probe_positive=bool(rng.random() < true_probe.likelihood[regime]),
                            raw_action=(),
                            transformed_action=(),
                            unsafe=bool(rng.random() < true_probe.unsafe_probability),
                        )
                    )

            for _ in range(action_trials_per_context):
                raw = np.asarray(env.action_space.sample(), dtype=float)
                transformed = adapter.transform_action(raw)
                records.append(
                    RawRecord(
                        context_id=context_id,
                        event_type="action",
                        probe=None,
                        probe_positive=None,
                        raw_action=tuple(float(x) for x in raw),
                        transformed_action=tuple(float(x) for x in transformed),
                        unsafe=adapter.is_unsafe_action(transformed),
                    )
                )
    finally:
        env.close()

    return RawRecordDataset(records=tuple(records), context_labels=context_labels)


def raw_records_to_fingerprints(dataset: RawRecordDataset) -> UnsupervisedDataset:
    """Derive context fingerprints from raw probe and action records."""
    by_context: dict[int, list[RawRecord]] = {}
    for record in dataset.records:
        by_context.setdefault(record.context_id, []).append(record)

    fingerprints: list[FingerprintSample] = []
    for context_id in sorted(by_context):
        records = by_context[context_id]
        true_regime = dataset.context_labels.get(context_id, "unknown")
        probe_rates = []
        for probe in PROBES:
            selected = [
                record
                for record in records
                if record.event_type == "probe" and record.probe == probe
            ]
            if not selected:
                probe_rates.append(0.0)
            else:
                probe_rates.append(
                    float(np.mean([bool(record.probe_positive) for record in selected]))
                )

        action_records = [record for record in records if record.event_type == "action"]
        raw_array = np.asarray([record.raw_action for record in action_records], dtype=float)
        transformed_array = np.asarray(
            [record.transformed_action for record in action_records],
            dtype=float,
        )
        unsafe_array = np.asarray([record.unsafe for record in action_records], dtype=bool)
        scale = estimate_action_scale(raw_array, transformed_array)
        noise = estimate_action_noise(raw_array, transformed_array, scale)
        threshold = estimate_unsafe_threshold(transformed_array, unsafe_array)
        fingerprints.append(
            FingerprintSample(
                true_regime=true_regime,
                probe_rates=tuple(float(x) for x in probe_rates),
                action_scale=scale,
                action_noise=noise,
                unsafe_threshold=threshold,
            )
        )

    return UnsupervisedDataset(fingerprints=tuple(fingerprints))


def sample_fingerprint(
    config: MetaWorldHiddenRegimeConfig,
    adapter: HiddenRegimeMetaWorldAdapter,
    rng: np.random.Generator,
    regime: str,
    probe_trials_per_probe: int,
    action_trials_per_context: int,
) -> FingerprintSample:
    probe_rates = []
    for probe in PROBES:
        true_probe = config.probes[probe]
        positives = sum(
            1
            for _ in range(probe_trials_per_probe)
            if rng.random() < true_probe.likelihood[regime]
        )
        probe_rates.append(positives / probe_trials_per_probe)

    raw_actions = []
    transformed_actions = []
    unsafe = []
    for _ in range(action_trials_per_context):
        raw = np.asarray(adapter.env.action_space.sample(), dtype=float)
        transformed = adapter.transform_action(raw)
        raw_actions.append(raw)
        transformed_actions.append(transformed)
        unsafe.append(adapter.is_unsafe_action(transformed))

    raw_array = np.asarray(raw_actions, dtype=float)
    transformed_array = np.asarray(transformed_actions, dtype=float)
    unsafe_array = np.asarray(unsafe, dtype=bool)
    scale = estimate_action_scale(raw_array, transformed_array)
    noise = estimate_action_noise(raw_array, transformed_array, scale)
    threshold = estimate_unsafe_threshold(transformed_array, unsafe_array)
    return FingerprintSample(
        true_regime=regime,
        probe_rates=tuple(float(x) for x in probe_rates),
        action_scale=scale,
        action_noise=noise,
        unsafe_threshold=threshold,
    )


def fit_learned_config(
    base_config: MetaWorldHiddenRegimeConfig,
    dataset: LearnedDataset,
    alpha: float = 1.0,
) -> tuple[MetaWorldHiddenRegimeConfig, dict[str, Any]]:
    probes = fit_probe_specs(base_config, dataset.probe_samples, alpha=alpha)
    regimes, diagnostics = fit_hidden_regimes(base_config, dataset.action_samples)
    learned_config = MetaWorldHiddenRegimeConfig(
        env_id=base_config.env_id,
        task=base_config.task,
        seed=base_config.seed,
        max_episode_steps=base_config.max_episode_steps,
        obstruction_threshold=base_config.obstruction_threshold,
        max_probes=base_config.max_probes,
        success_info_key=base_config.success_info_key,
        prior=tuple(float(x) for x in base_config.prior_array()),
        regimes=regimes,
        probes=probes,
    )
    diagnostics["probe_likelihood_mae"] = probe_likelihood_mae(base_config, learned_config)
    diagnostics["section_mae"] = section_mae(base_config, learned_config)
    return learned_config, diagnostics


def fit_unsupervised_config(
    base_config: MetaWorldHiddenRegimeConfig,
    dataset: UnsupervisedDataset,
    seed: int,
) -> tuple[MetaWorldHiddenRegimeConfig, dict[str, Any]]:
    features = np.asarray([sample.features() for sample in dataset.fingerprints], dtype=float)
    normalised, mean, std = standardise(features)
    assignments, centers = kmeans(normalised, k=len(REGIMES), seed=seed)
    cluster_order = order_clusters_by_section_signature(centers, mean, std)
    cluster_to_regime = {
        int(cluster): REGIMES[index]
        for index, cluster in enumerate(cluster_order)
    }

    probes: dict[str, ProbeSpec] = {}
    for probe_index, probe in enumerate(PROBES):
        likelihood: dict[str, float] = {}
        for cluster, regime_name in cluster_to_regime.items():
            cluster_values = [
                sample.probe_rates[probe_index]
                for sample, assignment in zip(dataset.fingerprints, assignments)
                if int(assignment) == cluster
            ]
            likelihood[regime_name] = float(np.clip(np.mean(cluster_values), 0.01, 0.99))
        true_probe = base_config.probes[probe]
        probes[probe] = ProbeSpec(
            name=probe,
            likelihood=likelihood,
            unsafe_probability=true_probe.unsafe_probability,
            reward_cost=true_probe.reward_cost,
        )

    regimes: dict[str, HiddenRegime] = {}
    for cluster, regime_name in cluster_to_regime.items():
        cluster_samples = [
            sample
            for sample, assignment in zip(dataset.fingerprints, assignments)
            if int(assignment) == cluster
        ]
        true_regime = base_config.regimes[regime_name]
        regimes[regime_name] = HiddenRegime(
            name=regime_name,
            action_scale=float(np.mean([sample.action_scale for sample in cluster_samples])),
            action_noise=float(np.mean([sample.action_noise for sample in cluster_samples])),
            unsafe_action_norm=float(np.mean([sample.unsafe_threshold for sample in cluster_samples])),
            unsafe_penalty=true_regime.unsafe_penalty,
        )

    learned_config = MetaWorldHiddenRegimeConfig(
        env_id=base_config.env_id,
        task=base_config.task,
        seed=base_config.seed,
        max_episode_steps=base_config.max_episode_steps,
        obstruction_threshold=base_config.obstruction_threshold,
        max_probes=base_config.max_probes,
        success_info_key=base_config.success_info_key,
        prior=tuple(float(x) for x in base_config.prior_array()),
        regimes=regimes,
        probes=probes,
    )
    diagnostics = {
        "cluster_to_regime": cluster_to_regime,
        "cluster_purity": cluster_purity(dataset, assignments),
        "probe_likelihood_mae": probe_likelihood_mae(base_config, learned_config),
        "section_mae": section_mae(base_config, learned_config),
    }
    return learned_config, diagnostics


def fit_probe_specs(
    base_config: MetaWorldHiddenRegimeConfig,
    samples: tuple[ProbeSample, ...],
    alpha: float,
) -> dict[str, ProbeSpec]:
    fitted: dict[str, ProbeSpec] = {}
    for probe in PROBES:
        likelihood: dict[str, float] = {}
        unsafe_count = 0
        probe_count = 0
        for regime in REGIMES:
            selected = [sample for sample in samples if sample.regime == regime and sample.probe == probe]
            positives = sum(1 for sample in selected if sample.positive)
            likelihood[regime] = float((positives + alpha) / (len(selected) + 2.0 * alpha))
            unsafe_count += sum(1 for sample in selected if sample.unsafe)
            probe_count += len(selected)
        true_probe = base_config.probes[probe]
        unsafe_probability = float((unsafe_count + alpha) / (probe_count + 2.0 * alpha))
        fitted[probe] = ProbeSpec(
            name=probe,
            likelihood=likelihood,
            unsafe_probability=unsafe_probability,
            reward_cost=true_probe.reward_cost,
        )
    return fitted


def fit_hidden_regimes(
    base_config: MetaWorldHiddenRegimeConfig,
    samples: tuple[ActionSample, ...],
) -> tuple[dict[str, HiddenRegime], dict[str, Any]]:
    regimes: dict[str, HiddenRegime] = {}
    diagnostics: dict[str, Any] = {"regimes": {}}
    for regime in REGIMES:
        selected = [sample for sample in samples if sample.regime == regime]
        raw = np.asarray([sample.raw_action for sample in selected], dtype=float)
        transformed = np.asarray([sample.transformed_action for sample in selected], dtype=float)
        unsafe = np.asarray([sample.unsafe for sample in selected], dtype=bool)
        scale = estimate_action_scale(raw, transformed)
        noise = estimate_action_noise(raw, transformed, scale)
        threshold = estimate_unsafe_threshold(transformed, unsafe)
        true_regime = base_config.regimes[regime]
        regimes[regime] = HiddenRegime(
            name=regime,
            action_scale=scale,
            action_noise=noise,
            unsafe_action_norm=threshold,
            unsafe_penalty=true_regime.unsafe_penalty,
        )
        diagnostics["regimes"][regime] = {
            "action_scale": scale,
            "action_noise": noise,
            "unsafe_action_norm": threshold,
            "unsafe_rate_in_training": float(np.mean(unsafe)),
        }
    return regimes, diagnostics


def estimate_action_scale(raw: np.ndarray, transformed: np.ndarray) -> float:
    denominator = float(np.sum(raw * raw))
    if denominator <= 1e-12:
        return 1.0
    scale = float(np.sum(raw * transformed) / denominator)
    return float(np.clip(scale, 0.05, 2.0))


def estimate_action_noise(raw: np.ndarray, transformed: np.ndarray, scale: float) -> float:
    residual = transformed - scale * raw
    return float(np.clip(np.std(residual), 0.0, 0.5))


def estimate_unsafe_threshold(transformed: np.ndarray, unsafe: np.ndarray) -> float:
    norms = np.linalg.norm(transformed, axis=1)
    if len(norms) == 0:
        return 1.0
    if not bool(np.any(unsafe)):
        return float(max(1.8, np.max(norms) * 1.05))
    if bool(np.all(unsafe)):
        return float(np.min(norms) * 0.95)
    candidates = np.linspace(float(np.min(norms)), float(np.max(norms)), num=128)
    best_threshold = float(np.median(norms))
    best_error = len(norms) + 1
    for threshold in candidates:
        predicted = norms > threshold
        error = int(np.sum(predicted != unsafe))
        if error < best_error:
            best_error = error
            best_threshold = float(threshold)
    return best_threshold


def probe_likelihood_mae(
    true_config: MetaWorldHiddenRegimeConfig,
    learned_config: MetaWorldHiddenRegimeConfig,
) -> float:
    errors = []
    for probe in PROBES:
        for regime in REGIMES:
            errors.append(
                abs(
                    true_config.probes[probe].likelihood[regime]
                    - learned_config.probes[probe].likelihood[regime]
                )
            )
    return float(np.mean(errors))


def section_mae(
    true_config: MetaWorldHiddenRegimeConfig,
    learned_config: MetaWorldHiddenRegimeConfig,
) -> float:
    errors = []
    for regime in REGIMES:
        true_regime = true_config.regimes[regime]
        learned_regime = learned_config.regimes[regime]
        errors.extend(
            [
                abs(true_regime.action_scale - learned_regime.action_scale),
                abs(true_regime.action_noise - learned_regime.action_noise),
                abs(true_regime.unsafe_action_norm - learned_regime.unsafe_action_norm),
            ]
        )
    return float(np.mean(errors))


def standardise(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    return (features - mean) / std, mean, std


def kmeans(
    features: np.ndarray,
    k: int,
    seed: int,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, len(features)))
    centers = [features[first]]
    while len(centers) < k:
        distances = np.min(
            np.stack([np.sum((features - center) ** 2, axis=1) for center in centers], axis=1),
            axis=1,
        )
        if float(distances.sum()) <= 1e-12:
            next_index = len(centers)
        else:
            probabilities = distances / distances.sum()
            next_index = int(rng.choice(np.arange(len(features)), p=probabilities))
        centers.append(features[next_index])
    centers_array = np.asarray(centers, dtype=float)
    assignments = np.zeros(len(features), dtype=int)
    for _ in range(max_iter):
        distances = np.stack(
            [np.sum((features - center) ** 2, axis=1) for center in centers_array],
            axis=1,
        )
        next_assignments = np.argmin(distances, axis=1)
        next_centers = centers_array.copy()
        for cluster in range(k):
            selected = features[next_assignments == cluster]
            if len(selected) > 0:
                next_centers[cluster] = selected.mean(axis=0)
        if np.array_equal(assignments, next_assignments):
            centers_array = next_centers
            break
        assignments = next_assignments
        centers_array = next_centers
    return assignments, centers_array


def order_clusters_by_section_signature(
    centers: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> list[int]:
    raw_centers = centers * std + mean
    cluster_indices = list(range(len(raw_centers)))
    slippery = max(cluster_indices, key=lambda i: raw_centers[i, 0])
    remaining = [i for i in cluster_indices if i != slippery]
    fragile = max(remaining, key=lambda i: raw_centers[i, 1])
    remaining = [i for i in remaining if i != fragile]
    heavy = max(remaining, key=lambda i: raw_centers[i, 2])
    nominal = [i for i in remaining if i != heavy][0]
    return [nominal, slippery, fragile, heavy]


def cluster_purity(dataset: UnsupervisedDataset, assignments: np.ndarray) -> float:
    correct = 0
    for cluster in sorted(set(int(x) for x in assignments)):
        labels = [
            sample.true_regime
            for sample, assignment in zip(dataset.fingerprints, assignments)
            if int(assignment) == cluster
        ]
        if labels:
            correct += max(labels.count(label) for label in set(labels))
    return float(correct / len(assignments))


def run_learned_strategy_benchmark(
    config: MetaWorldHiddenRegimeConfig,
    train_probe_samples_per_regime_probe: int,
    train_action_samples_per_regime: int,
    eval_episodes: int,
    seed: int,
    strategy_names: list[str] | None = None,
) -> dict[str, Any]:
    dataset = collect_learning_data(
        config=config,
        probe_samples_per_regime_probe=train_probe_samples_per_regime_probe,
        action_samples_per_regime=train_action_samples_per_regime,
        seed=seed,
    )
    learned_config, diagnostics = fit_learned_config(config, dataset)
    evaluation = run_strategy_benchmark(
        config=config,
        episodes=eval_episodes,
        strategy_names=strategy_names,
        model_config=learned_config,
    )
    return {
        "benchmark": "metaworld_hidden_regime_learned_model",
        "training": {
            "seed": seed,
            "probe_samples_per_regime_probe": train_probe_samples_per_regime_probe,
            "action_samples_per_regime": train_action_samples_per_regime,
            "dataset": dataset.as_dict(),
        },
        "fit_diagnostics": diagnostics,
        "learned_model": learned_config.as_dict(),
        "evaluation": evaluation,
    }


def run_unsupervised_strategy_benchmark(
    config: MetaWorldHiddenRegimeConfig,
    contexts_per_regime: int,
    probe_trials_per_probe: int,
    action_trials_per_context: int,
    eval_episodes: int,
    seed: int,
    strategy_names: list[str] | None = None,
) -> dict[str, Any]:
    dataset = collect_unlabeled_fingerprints(
        config=config,
        contexts_per_regime=contexts_per_regime,
        probe_trials_per_probe=probe_trials_per_probe,
        action_trials_per_context=action_trials_per_context,
        seed=seed,
    )
    learned_config, diagnostics = fit_unsupervised_config(config, dataset, seed=seed)
    evaluation = run_strategy_benchmark(
        config=config,
        episodes=eval_episodes,
        strategy_names=strategy_names,
        model_config=learned_config,
    )
    return {
        "benchmark": "metaworld_hidden_regime_unsupervised_model",
        "training": {
            "seed": seed,
            "contexts_per_regime": contexts_per_regime,
            "probe_trials_per_probe": probe_trials_per_probe,
            "action_trials_per_context": action_trials_per_context,
            "dataset": dataset.as_dict(),
        },
        "fit_diagnostics": diagnostics,
        "learned_model": learned_config.as_dict(),
        "evaluation": evaluation,
    }


def run_stream_unsupervised_strategy_benchmark(
    config: MetaWorldHiddenRegimeConfig,
    total_contexts: int,
    probe_trials_per_probe: int,
    action_trials_per_context: int,
    eval_episodes: int,
    seed: int,
    strategy_names: list[str] | None = None,
) -> dict[str, Any]:
    dataset = collect_unlabeled_stream_fingerprints(
        config=config,
        total_contexts=total_contexts,
        probe_trials_per_probe=probe_trials_per_probe,
        action_trials_per_context=action_trials_per_context,
        seed=seed,
    )
    learned_config, diagnostics = fit_unsupervised_config(config, dataset, seed=seed)
    evaluation = run_strategy_benchmark(
        config=config,
        episodes=eval_episodes,
        strategy_names=strategy_names,
        model_config=learned_config,
    )
    return {
        "benchmark": "metaworld_hidden_regime_stream_unsupervised_model",
        "training": {
            "seed": seed,
            "total_contexts": total_contexts,
            "probe_trials_per_probe": probe_trials_per_probe,
            "action_trials_per_context": action_trials_per_context,
            "dataset": dataset.as_dict(),
        },
        "fit_diagnostics": diagnostics,
        "learned_model": learned_config.as_dict(),
        "evaluation": evaluation,
    }


def run_raw_record_strategy_benchmark(
    config: MetaWorldHiddenRegimeConfig,
    total_contexts: int,
    probe_trials_per_probe: int,
    action_trials_per_context: int,
    eval_episodes: int,
    seed: int,
    strategy_names: list[str] | None = None,
) -> dict[str, Any]:
    raw_dataset = collect_unlabeled_raw_stream_records(
        config=config,
        total_contexts=total_contexts,
        probe_trials_per_probe=probe_trials_per_probe,
        action_trials_per_context=action_trials_per_context,
        seed=seed,
    )
    dataset = raw_records_to_fingerprints(raw_dataset)
    learned_config, diagnostics = fit_unsupervised_config(config, dataset, seed=seed)
    evaluation = run_strategy_benchmark(
        config=config,
        episodes=eval_episodes,
        strategy_names=strategy_names,
        model_config=learned_config,
    )
    return {
        "benchmark": "metaworld_hidden_regime_raw_record_model",
        "training": {
            "seed": seed,
            "total_contexts": total_contexts,
            "probe_trials_per_probe": probe_trials_per_probe,
            "action_trials_per_context": action_trials_per_context,
            "raw_dataset": raw_dataset.as_dict(),
            "derived_dataset": dataset.as_dict(),
        },
        "fit_diagnostics": diagnostics,
        "learned_model": learned_config.as_dict(),
        "evaluation": evaluation,
    }


def _count_probe_samples(samples: tuple[ProbeSample, ...]) -> dict[str, dict[str, int]]:
    counts = {probe: {regime: 0 for regime in REGIMES} for probe in PROBES}
    for sample in samples:
        counts[sample.probe][sample.regime] += 1
    return counts


def _count_action_samples(samples: tuple[ActionSample, ...]) -> dict[str, int]:
    counts = {regime: 0 for regime in REGIMES}
    for sample in samples:
        counts[sample.regime] += 1
    return counts


def _count_fingerprints(samples: tuple[FingerprintSample, ...]) -> dict[str, int]:
    counts = {regime: 0 for regime in REGIMES}
    for sample in samples:
        counts[sample.true_regime] += 1
    return counts


def _count_raw_events(samples: tuple[RawRecord, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        counts[sample.event_type] = counts.get(sample.event_type, 0) + 1
    return counts


def _count_raw_contexts(context_labels: dict[int, str]) -> dict[str, int]:
    counts = {regime: 0 for regime in REGIMES}
    for regime in context_labels.values():
        counts[regime] += 1
    return counts
