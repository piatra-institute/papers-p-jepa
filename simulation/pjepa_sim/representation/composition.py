"""Skill-composition benchmark for action-grounded representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pjepa_sim.representation.clustering import assign_nearest, best_kmeans, standardise
from pjepa_sim.representation.learning import (
    REGIMES,
    TEST_VISUAL,
    TRAIN_VISUAL,
    VISUALS,
    cluster_purity,
    visual_features,
)


PREP_SKILLS = ("no_prep", "wipe", "cushion", "brace")
FINISH_SKILLS = ("fast_lift", "slow_lift", "grip_hard", "two_contact_lift")
POSTCONDITIONS = (
    "dry_ready",
    "slip_ready",
    "fragile_ready",
    "heavy_ready",
    "clean_ready",
    "protected_ready",
    "supported_ready",
)


@dataclass(frozen=True)
class PrepOutcome:
    postcondition: str
    success: float
    unsafe: float


@dataclass(frozen=True)
class FinishOutcome:
    success: float
    unsafe: float


@dataclass(frozen=True)
class ChainOutcome:
    success: float
    unsafe: float


@dataclass(frozen=True)
class SkillRecord:
    split: str
    context_id: int
    regime: str
    visual: str
    visual_features: tuple[float, ...]
    composition_features: tuple[float, ...]


@dataclass(frozen=True)
class CompositionResult:
    name: str
    success_rate: float
    unsafe_failure_rate: float
    risk_adjusted_score: float
    cluster_purity: float
    by_regime: dict[str, dict[str, float | str]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "success_rate": self.success_rate,
            "unsafe_failure_rate": self.unsafe_failure_rate,
            "risk_adjusted_score": self.risk_adjusted_score,
            "cluster_purity": self.cluster_purity,
            "by_regime": self.by_regime,
        }


PREP_MODEL: dict[str, dict[str, PrepOutcome]] = {
    "dry": {
        "no_prep": PrepOutcome("dry_ready", 0.99, 0.00),
        "wipe": PrepOutcome("dry_ready", 0.96, 0.01),
        "cushion": PrepOutcome("dry_ready", 0.94, 0.01),
        "brace": PrepOutcome("dry_ready", 0.94, 0.01),
    },
    "soapy": {
        "no_prep": PrepOutcome("slip_ready", 0.99, 0.00),
        "wipe": PrepOutcome("clean_ready", 0.94, 0.01),
        "cushion": PrepOutcome("slip_ready", 0.92, 0.03),
        "brace": PrepOutcome("slip_ready", 0.90, 0.03),
    },
    "cracked": {
        "no_prep": PrepOutcome("fragile_ready", 0.99, 0.00),
        "wipe": PrepOutcome("fragile_ready", 0.90, 0.04),
        "cushion": PrepOutcome("protected_ready", 0.93, 0.01),
        "brace": PrepOutcome("fragile_ready", 0.88, 0.05),
    },
    "heavy": {
        "no_prep": PrepOutcome("heavy_ready", 0.99, 0.00),
        "wipe": PrepOutcome("heavy_ready", 0.91, 0.03),
        "cushion": PrepOutcome("heavy_ready", 0.90, 0.03),
        "brace": PrepOutcome("supported_ready", 0.94, 0.01),
    },
}


FINISH_MODEL: dict[str, dict[str, FinishOutcome]] = {
    "dry_ready": {
        "fast_lift": FinishOutcome(0.96, 0.02),
        "slow_lift": FinishOutcome(0.88, 0.01),
        "grip_hard": FinishOutcome(0.94, 0.02),
        "two_contact_lift": FinishOutcome(0.84, 0.01),
    },
    "slip_ready": {
        "fast_lift": FinishOutcome(0.18, 0.75),
        "slow_lift": FinishOutcome(0.62, 0.28),
        "grip_hard": FinishOutcome(0.42, 0.48),
        "two_contact_lift": FinishOutcome(0.76, 0.18),
    },
    "fragile_ready": {
        "fast_lift": FinishOutcome(0.36, 0.52),
        "slow_lift": FinishOutcome(0.86, 0.08),
        "grip_hard": FinishOutcome(0.12, 0.84),
        "two_contact_lift": FinishOutcome(0.55, 0.35),
    },
    "heavy_ready": {
        "fast_lift": FinishOutcome(0.30, 0.60),
        "slow_lift": FinishOutcome(0.48, 0.42),
        "grip_hard": FinishOutcome(0.74, 0.20),
        "two_contact_lift": FinishOutcome(0.60, 0.25),
    },
    "clean_ready": {
        "fast_lift": FinishOutcome(0.85, 0.08),
        "slow_lift": FinishOutcome(0.82, 0.04),
        "grip_hard": FinishOutcome(0.80, 0.06),
        "two_contact_lift": FinishOutcome(0.92, 0.03),
    },
    "protected_ready": {
        "fast_lift": FinishOutcome(0.58, 0.30),
        "slow_lift": FinishOutcome(0.93, 0.03),
        "grip_hard": FinishOutcome(0.50, 0.36),
        "two_contact_lift": FinishOutcome(0.80, 0.08),
    },
    "supported_ready": {
        "fast_lift": FinishOutcome(0.70, 0.16),
        "slow_lift": FinishOutcome(0.74, 0.12),
        "grip_hard": FinishOutcome(0.95, 0.02),
        "two_contact_lift": FinishOutcome(0.90, 0.04),
    },
}


def run_skill_composition_benchmark(
    contexts_per_regime: int = 96,
    feature_noise: float = 0.035,
    unsafe_weight: float = 2.0,
    prep_cost: float = 0.015,
    seed: int = 11,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    train = generate_records("train", contexts_per_regime, feature_noise, rng)
    test = generate_records("test", contexts_per_regime, feature_noise, rng)
    learners = {
        "prior_average": evaluate_prior(train, test, unsafe_weight, prep_cost),
        "appearance_grouping": evaluate_appearance(train, test, unsafe_weight, prep_cost),
        "action_consequence_grouping": evaluate_action_consequence(
            train,
            test,
            unsafe_weight,
            prep_cost,
            seed,
        ),
        "oracle_regime": evaluate_oracle(test, unsafe_weight, prep_cost),
    }
    return {
        "benchmark": "skill_composition",
        "description": (
            "A two-step task where the first skill creates an intermediate "
            "postcondition and the second skill must be valid for that postcondition."
        ),
        "config": {
            "regimes": list(REGIMES),
            "prep_skills": list(PREP_SKILLS),
            "finish_skills": list(FINISH_SKILLS),
            "postconditions": list(POSTCONDITIONS),
            "train_visual_mapping": TRAIN_VISUAL,
            "test_visual_mapping": TEST_VISUAL,
            "contexts_per_regime": contexts_per_regime,
            "feature_noise": feature_noise,
            "unsafe_weight": unsafe_weight,
            "prep_cost": prep_cost,
            "seed": seed,
        },
        "learners": {name: result.as_dict() for name, result in learners.items()},
    }


def generate_records(
    split: str,
    contexts_per_regime: int,
    feature_noise: float,
    rng: np.random.Generator,
) -> tuple[SkillRecord, ...]:
    visual_map = TRAIN_VISUAL if split == "train" else TEST_VISUAL
    records: list[SkillRecord] = []
    context_id = 0
    for regime in REGIMES:
        for _ in range(contexts_per_regime):
            visual = visual_map[regime]
            records.append(
                SkillRecord(
                    split=split,
                    context_id=context_id,
                    regime=regime,
                    visual=visual,
                    visual_features=visual_features(visual),
                    composition_features=composition_features(regime, feature_noise, rng),
                )
            )
            context_id += 1
    return tuple(records)


def composition_features(
    regime: str,
    feature_noise: float,
    rng: np.random.Generator,
) -> tuple[float, ...]:
    values = []
    for prep in PREP_SKILLS:
        outcome = PREP_MODEL[regime][prep]
        postcondition = [1.0 if outcome.postcondition == item else 0.0 for item in POSTCONDITIONS]
        values.extend((outcome.success, outcome.unsafe, *postcondition))
    for prep in PREP_SKILLS:
        for finish in FINISH_SKILLS:
            outcome = evaluate_chain(regime, prep, finish)
            values.extend((outcome.success, outcome.unsafe))
    noisy = np.asarray(values, dtype=float) + rng.normal(0.0, feature_noise, size=len(values))
    return tuple(float(np.clip(value, 0.0, 1.0)) for value in noisy)


def evaluate_prior(
    train: tuple[SkillRecord, ...],
    test: tuple[SkillRecord, ...],
    unsafe_weight: float,
    prep_cost: float,
) -> CompositionResult:
    train_assignments = np.zeros(len(train), dtype=int)
    test_assignments = np.zeros(len(test), dtype=int)
    sections = fit_sections(train, train_assignments)
    return evaluate_assigned_model(
        "prior_average",
        test,
        test_assignments,
        sections,
        cluster_purity(test, test_assignments),
        unsafe_weight,
        prep_cost,
    )


def evaluate_appearance(
    train: tuple[SkillRecord, ...],
    test: tuple[SkillRecord, ...],
    unsafe_weight: float,
    prep_cost: float,
) -> CompositionResult:
    visual_to_cluster = {visual: index for index, visual in enumerate(VISUALS)}
    train_assignments = np.asarray([visual_to_cluster[record.visual] for record in train], dtype=int)
    test_assignments = np.asarray([visual_to_cluster[record.visual] for record in test], dtype=int)
    sections = fit_sections(train, train_assignments)
    return evaluate_assigned_model(
        "appearance_grouping",
        test,
        test_assignments,
        sections,
        cluster_purity(test, test_assignments),
        unsafe_weight,
        prep_cost,
    )


def evaluate_action_consequence(
    train: tuple[SkillRecord, ...],
    test: tuple[SkillRecord, ...],
    unsafe_weight: float,
    prep_cost: float,
    seed: int,
) -> CompositionResult:
    train_features = np.asarray([record.composition_features for record in train], dtype=float)
    test_features = np.asarray([record.composition_features for record in test], dtype=float)
    train_norm, mean, std = standardise(train_features)
    test_norm = (test_features - mean) / std
    train_assignments, centers = best_kmeans(train_norm, k=len(REGIMES), seed=seed)
    test_assignments = assign_nearest(test_norm, centers)
    sections = fit_sections(train, train_assignments)
    return evaluate_assigned_model(
        "action_consequence_grouping",
        test,
        test_assignments,
        sections,
        cluster_purity(test, test_assignments),
        unsafe_weight,
        prep_cost,
    )


def evaluate_oracle(
    test: tuple[SkillRecord, ...],
    unsafe_weight: float,
    prep_cost: float,
) -> CompositionResult:
    regime_to_cluster = {regime: index for index, regime in enumerate(REGIMES)}
    assignments = np.asarray([regime_to_cluster[record.regime] for record in test], dtype=int)
    sections = {
        regime_to_cluster[regime]: {
            prep: {
                finish: evaluate_chain(regime, prep, finish)
                for finish in FINISH_SKILLS
            }
            for prep in PREP_SKILLS
        }
        for regime in REGIMES
    }
    return evaluate_assigned_model(
        "oracle_regime",
        test,
        assignments,
        sections,
        1.0,
        unsafe_weight,
        prep_cost,
    )


def fit_sections(
    records: tuple[SkillRecord, ...],
    assignments: np.ndarray,
) -> dict[int, dict[str, dict[str, ChainOutcome]]]:
    sections: dict[int, dict[str, dict[str, ChainOutcome]]] = {}
    for cluster in sorted(set(int(value) for value in assignments)):
        selected = [record for record, assignment in zip(records, assignments) if int(assignment) == cluster]
        sections[cluster] = {}
        for prep in PREP_SKILLS:
            sections[cluster][prep] = {}
            for finish in FINISH_SKILLS:
                outcomes = [evaluate_chain(record.regime, prep, finish) for record in selected]
                sections[cluster][prep][finish] = ChainOutcome(
                    success=float(np.mean([outcome.success for outcome in outcomes])),
                    unsafe=float(np.mean([outcome.unsafe for outcome in outcomes])),
                )
    return sections


def evaluate_assigned_model(
    name: str,
    test: tuple[SkillRecord, ...],
    assignments: np.ndarray,
    sections: dict[int, dict[str, dict[str, ChainOutcome]]],
    purity: float,
    unsafe_weight: float,
    prep_cost: float,
) -> CompositionResult:
    total_success = 0.0
    total_unsafe = 0.0
    by_regime: dict[str, dict[str, float | str]] = {}
    for regime in REGIMES:
        regime_records = [record for record in test if record.regime == regime]
        success = 0.0
        unsafe = 0.0
        chain_counts: dict[str, int] = {}
        for record in regime_records:
            assignment = int(assignments[record.context_id])
            prep, finish = best_chain(sections[assignment], unsafe_weight, prep_cost)
            chain_counts[f"{prep}->{finish}"] = chain_counts.get(f"{prep}->{finish}", 0) + 1
            outcome = evaluate_chain(record.regime, prep, finish)
            success += outcome.success
            unsafe += outcome.unsafe
        total_success += success
        total_unsafe += unsafe
        n_regime = float(len(regime_records))
        by_regime[regime] = {
            "dominant_chain": max(chain_counts, key=chain_counts.get),
            "success_rate": success / n_regime,
            "unsafe_failure_rate": unsafe / n_regime,
        }

    n_total = float(len(test))
    success_rate = total_success / n_total
    unsafe_rate = total_unsafe / n_total
    return CompositionResult(
        name=name,
        success_rate=success_rate,
        unsafe_failure_rate=unsafe_rate,
        risk_adjusted_score=success_rate - unsafe_weight * unsafe_rate,
        cluster_purity=purity,
        by_regime=by_regime,
    )


def best_chain(
    section: dict[str, dict[str, ChainOutcome]],
    unsafe_weight: float,
    prep_cost: float,
) -> tuple[str, str]:
    candidates = []
    for prep, finish_map in section.items():
        for finish, outcome in finish_map.items():
            cost = 0.0 if prep == "no_prep" else prep_cost
            score = outcome.success - unsafe_weight * outcome.unsafe - cost
            candidates.append((score, prep, finish))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    _, prep, finish = candidates[0]
    return prep, finish


def evaluate_chain(regime: str, prep: str, finish: str) -> ChainOutcome:
    prep_outcome = PREP_MODEL[regime][prep]
    finish_outcome = FINISH_MODEL[prep_outcome.postcondition][finish]
    success = prep_outcome.success * finish_outcome.success
    unsafe = prep_outcome.unsafe + (1.0 - prep_outcome.unsafe) * finish_outcome.unsafe
    return ChainOutcome(success=success, unsafe=unsafe)

