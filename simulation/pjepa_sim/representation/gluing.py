"""Restriction-map ablation for local action-model gluing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pjepa_sim.core.dishworld import ACTION_MODEL, DIRECT_ACTIONS, REGIMES


CHARTS = ("vision", "contact", "load")
REFERENCE_CHART = "vision"
CHART_PERMUTATIONS: dict[str, tuple[int, ...]] = {
    "vision": (0, 1, 2, 3),
    "contact": (2, 3, 1, 0),
    "load": (3, 0, 2, 1),
}
CHART_NOISE: dict[str, float] = {
    "vision": 0.22,
    "contact": 0.14,
    "load": 0.14,
}


@dataclass(frozen=True)
class GluingRecord:
    split: str
    context_id: int
    regime: str
    charts: dict[str, tuple[float, ...]]


@dataclass(frozen=True)
class RestrictionMap:
    matrix: np.ndarray
    bias: np.ndarray

    def apply(self, values: np.ndarray) -> np.ndarray:
        return values @ self.matrix + self.bias


@dataclass(frozen=True)
class GluingResult:
    name: str
    success_rate: float
    unsafe_failure_rate: float
    risk_adjusted_score: float
    mean_overlap_residual: float
    by_regime: dict[str, dict[str, float | str]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "success_rate": self.success_rate,
            "unsafe_failure_rate": self.unsafe_failure_rate,
            "risk_adjusted_score": self.risk_adjusted_score,
            "mean_overlap_residual": self.mean_overlap_residual,
            "by_regime": self.by_regime,
        }


def run_gluing_ablation_benchmark(
    train_contexts_per_regime: int = 256,
    test_contexts_per_regime: int = 512,
    latent_jitter: float = 0.04,
    unsafe_weight: float = 2.0,
    seed: int = 3,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    train = generate_records("train", train_contexts_per_regime, latent_jitter, unsafe_weight, rng)
    test = generate_records("test", test_contexts_per_regime, latent_jitter, unsafe_weight, rng)
    learned_maps = fit_restriction_maps(train)
    learners = {
        "reference_only": evaluate_reference_only(test, unsafe_weight),
        "identity_no_glue": evaluate_with_maps(test, identity_maps(), unsafe_weight, "identity_no_glue"),
        "learned_restriction_glue": evaluate_with_maps(test, learned_maps, unsafe_weight, "learned_restriction_glue"),
        "oracle_restriction_glue": evaluate_with_maps(test, oracle_maps(), unsafe_weight, "oracle_restriction_glue"),
        "oracle_regime": evaluate_oracle_regime(test, unsafe_weight),
    }
    return {
        "benchmark": "restriction_map_gluing_ablation",
        "description": (
            "Local action sections use incompatible action-coordinate frames. "
            "The benchmark compares identity/no-glue aggregation with restriction maps learned from unlabeled overlap records."
        ),
        "config": {
            "regimes": list(REGIMES),
            "direct_actions": list(DIRECT_ACTIONS),
            "charts": list(CHARTS),
            "reference_chart": REFERENCE_CHART,
            "chart_permutations": {name: list(permutation) for name, permutation in CHART_PERMUTATIONS.items()},
            "chart_noise": CHART_NOISE,
            "train_contexts_per_regime": train_contexts_per_regime,
            "test_contexts_per_regime": test_contexts_per_regime,
            "latent_jitter": latent_jitter,
            "unsafe_weight": unsafe_weight,
            "seed": seed,
        },
        "learners": {name: result.as_dict() for name, result in learners.items()},
    }


def generate_records(
    split: str,
    contexts_per_regime: int,
    latent_jitter: float,
    unsafe_weight: float,
    rng: np.random.Generator,
) -> tuple[GluingRecord, ...]:
    records: list[GluingRecord] = []
    context_id = 0
    for regime in REGIMES:
        base = canonical_utility(regime, unsafe_weight)
        for _ in range(contexts_per_regime):
            latent = base + rng.normal(0.0, latent_jitter, size=len(DIRECT_ACTIONS))
            charts = {}
            for chart in CHARTS:
                permutation = CHART_PERMUTATIONS[chart]
                local_values = latent[list(permutation)] + rng.normal(
                    0.0,
                    CHART_NOISE[chart],
                    size=len(DIRECT_ACTIONS),
                )
                charts[chart] = tuple(float(value) for value in local_values)
            records.append(
                GluingRecord(
                    split=split,
                    context_id=context_id,
                    regime=regime,
                    charts=charts,
                )
            )
            context_id += 1
    return tuple(records)


def canonical_utility(regime: str, unsafe_weight: float) -> np.ndarray:
    return np.asarray(
        [
            ACTION_MODEL[regime][action].success - unsafe_weight * ACTION_MODEL[regime][action].unsafe
            for action in DIRECT_ACTIONS
        ],
        dtype=float,
    )


def fit_restriction_maps(records: tuple[GluingRecord, ...]) -> dict[str, RestrictionMap]:
    target = np.asarray([record.charts[REFERENCE_CHART] for record in records], dtype=float)
    maps = {REFERENCE_CHART: RestrictionMap(np.eye(len(DIRECT_ACTIONS)), np.zeros(len(DIRECT_ACTIONS)))}
    for chart in CHARTS:
        if chart == REFERENCE_CHART:
            continue
        source = np.asarray([record.charts[chart] for record in records], dtype=float)
        source_augmented = np.c_[source, np.ones(len(source))]
        weights = np.linalg.lstsq(source_augmented, target, rcond=None)[0]
        maps[chart] = RestrictionMap(
            matrix=weights[:-1, :],
            bias=weights[-1, :],
        )
    return maps


def identity_maps() -> dict[str, RestrictionMap]:
    return {
        chart: RestrictionMap(np.eye(len(DIRECT_ACTIONS)), np.zeros(len(DIRECT_ACTIONS)))
        for chart in CHARTS
    }


def oracle_maps() -> dict[str, RestrictionMap]:
    maps = {}
    for chart, permutation in CHART_PERMUTATIONS.items():
        matrix = np.zeros((len(DIRECT_ACTIONS), len(DIRECT_ACTIONS)))
        for local_index, canonical_index in enumerate(permutation):
            matrix[local_index, canonical_index] = 1.0
        maps[chart] = RestrictionMap(matrix=matrix, bias=np.zeros(len(DIRECT_ACTIONS)))
    return maps


def evaluate_reference_only(
    records: tuple[GluingRecord, ...],
    unsafe_weight: float,
) -> GluingResult:
    return evaluate_policy(
        records,
        unsafe_weight,
        name="reference_only",
        predictor=lambda record: np.asarray(record.charts[REFERENCE_CHART], dtype=float),
        residual_fn=lambda record: 0.0,
    )


def evaluate_with_maps(
    records: tuple[GluingRecord, ...],
    maps: dict[str, RestrictionMap],
    unsafe_weight: float,
    name: str,
) -> GluingResult:
    return evaluate_policy(
        records,
        unsafe_weight,
        name=name,
        predictor=lambda record: aggregate_mapped_sections(record, maps),
        residual_fn=lambda record: overlap_residual(record, maps),
    )


def evaluate_oracle_regime(
    records: tuple[GluingRecord, ...],
    unsafe_weight: float,
) -> GluingResult:
    return evaluate_policy(
        records,
        unsafe_weight,
        name="oracle_regime",
        predictor=lambda record: canonical_utility(record.regime, unsafe_weight),
        residual_fn=lambda record: 0.0,
    )


def aggregate_mapped_sections(
    record: GluingRecord,
    maps: dict[str, RestrictionMap],
) -> np.ndarray:
    mapped = []
    for chart in CHARTS:
        values = np.asarray(record.charts[chart], dtype=float)
        mapped.append(maps[chart].apply(values))
    return np.asarray(mapped, dtype=float).mean(axis=0)


def overlap_residual(
    record: GluingRecord,
    maps: dict[str, RestrictionMap],
) -> float:
    mapped = []
    for chart in CHARTS:
        values = np.asarray(record.charts[chart], dtype=float)
        mapped.append(maps[chart].apply(values))
    matrix = np.asarray(mapped, dtype=float)
    mean = matrix.mean(axis=0)
    return float(np.mean((matrix - mean) ** 2))


def evaluate_policy(
    records: tuple[GluingRecord, ...],
    unsafe_weight: float,
    name: str,
    predictor,
    residual_fn,
) -> GluingResult:
    total_success = 0.0
    total_unsafe = 0.0
    total_residual = 0.0
    by_regime: dict[str, dict[str, float | str]] = {}

    for regime in REGIMES:
        selected = [record for record in records if record.regime == regime]
        success = 0.0
        unsafe = 0.0
        residual = 0.0
        action_counts: dict[str, int] = {}
        for record in selected:
            prediction = predictor(record)
            action = DIRECT_ACTIONS[int(np.argmax(prediction))]
            action_counts[action] = action_counts.get(action, 0) + 1
            outcome = ACTION_MODEL[record.regime][action]
            success += outcome.success
            unsafe += outcome.unsafe
            residual += residual_fn(record)
        total_success += success
        total_unsafe += unsafe
        total_residual += residual
        n_regime = float(len(selected))
        by_regime[regime] = {
            "dominant_action": max(action_counts, key=action_counts.get),
            "success_rate": success / n_regime,
            "unsafe_failure_rate": unsafe / n_regime,
            "mean_overlap_residual": residual / n_regime,
        }

    n_total = float(len(records))
    success_rate = total_success / n_total
    unsafe_rate = total_unsafe / n_total
    return GluingResult(
        name=name,
        success_rate=success_rate,
        unsafe_failure_rate=unsafe_rate,
        risk_adjusted_score=success_rate - unsafe_weight * unsafe_rate,
        mean_overlap_residual=total_residual / n_total,
        by_regime=by_regime,
    )
