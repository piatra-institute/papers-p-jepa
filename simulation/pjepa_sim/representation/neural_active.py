"""Learned active-probing benchmark under ambiguous sensor observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pjepa_sim.core.dishworld import ACTION_MODEL, DIRECT_ACTIONS, PROBE_LIKELIHOOD, PROBE_UNSAFE, PROBES, REGIMES
from pjepa_sim.representation.neural import TESTS, TinyMLP


UNKNOWN_EVIDENCE = 0.5
SensorBase = dict[str, tuple[float, ...]]
ProbeLikelihood = dict[str, dict[str, float]]
ProbeUnsafe = dict[str, float]

ACTIVE_SENSOR_BASE: dict[str, tuple[float, ...]] = {
    "dry": (0.15, 0.15),
    "soapy": (0.15, 0.15),
    "cracked": (0.85, 0.85),
    "heavy": (0.85, 0.85),
}
DISTINCT_SENSOR_BASE: dict[str, tuple[float, ...]] = {
    "dry": (0.12, 0.12),
    "soapy": (0.12, 0.88),
    "cracked": (0.88, 0.12),
    "heavy": (0.88, 0.88),
}


@dataclass(frozen=True)
class ActiveContext:
    context_id: int
    regime: str
    sensor_features: tuple[float, ...]


@dataclass(frozen=True)
class ActiveRecord:
    context_id: int
    regime: str
    sensor_features: tuple[float, ...]
    evidence_features: tuple[float, ...]
    test: str
    target: tuple[float, float]


@dataclass(frozen=True)
class ActiveResult:
    name: str
    success_rate: float
    unsafe_failure_rate: float
    mean_probes: float
    risk_adjusted_score: float
    by_regime: dict[str, dict[str, float | str]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "success_rate": self.success_rate,
            "unsafe_failure_rate": self.unsafe_failure_rate,
            "mean_probes": self.mean_probes,
            "risk_adjusted_score": self.risk_adjusted_score,
            "by_regime": self.by_regime,
        }


@dataclass(frozen=True)
class BoundarySpec:
    description: str
    sensor_base: SensorBase
    probe_likelihood: ProbeLikelihood
    probe_weight: float


def run_neural_active_probe_benchmark(
    contexts_per_regime: int = 96,
    intervention_repeats: int = 4,
    evidence_repeats: int = 2,
    sensor_noise: float = 0.020,
    hidden_dim: int = 32,
    max_probes: int = 2,
    unsafe_weight: float = 2.0,
    probe_weight: float = 0.02,
    seed: int = 53,
    sensor_base: SensorBase | None = None,
    probe_likelihood: ProbeLikelihood | None = None,
    probe_unsafe: ProbeUnsafe | None = None,
) -> dict[str, Any]:
    sensor_base = sensor_base or ACTIVE_SENSOR_BASE
    probe_likelihood = probe_likelihood or PROBE_LIKELIHOOD
    probe_unsafe = probe_unsafe or PROBE_UNSAFE
    rng = np.random.default_rng(seed)
    train_contexts = generate_active_contexts(contexts_per_regime, sensor_noise, rng, sensor_base)
    test_contexts = generate_active_contexts(contexts_per_regime, sensor_noise, rng, sensor_base)
    train_records = generate_active_records(
        train_contexts,
        intervention_repeats,
        evidence_repeats,
        rng,
        probe_likelihood,
        probe_unsafe,
    )
    model = train_active_model(train_records, hidden_dim, seed + 1)
    learners = {
        "learned_no_probe": evaluate_policy(
            "learned_no_probe",
            test_contexts,
            model,
            max_probes=0,
            unsafe_weight=unsafe_weight,
            probe_weight=probe_weight,
            probe_likelihood=probe_likelihood,
            probe_unsafe=probe_unsafe,
        ),
        "learned_entropy_probe": evaluate_policy(
            "learned_entropy_probe",
            test_contexts,
            model,
            max_probes=max_probes,
            unsafe_weight=unsafe_weight,
            probe_weight=probe_weight,
            entropy_only=True,
            probe_likelihood=probe_likelihood,
            probe_unsafe=probe_unsafe,
        ),
        "learned_active_probe": evaluate_policy(
            "learned_active_probe",
            test_contexts,
            model,
            max_probes=max_probes,
            unsafe_weight=unsafe_weight,
            probe_weight=probe_weight,
            probe_likelihood=probe_likelihood,
            probe_unsafe=probe_unsafe,
        ),
        "oracle_regime": evaluate_oracle(test_contexts, unsafe_weight, probe_weight),
    }
    return {
        "benchmark": "neural_active_probe",
        "description": (
            "Initial physical sensor observations alias pairs of hidden regimes. "
            "A NumPy MLP trained from intervention and probe-evidence records chooses safe probes before acting."
        ),
        "config": {
            "regimes": list(REGIMES),
            "direct_actions": list(DIRECT_ACTIONS),
            "probes": list(PROBES),
            "sensor_base": sensor_base,
            "probe_likelihood": probe_likelihood,
            "probe_unsafe": probe_unsafe,
            "contexts_per_regime": contexts_per_regime,
            "intervention_repeats": intervention_repeats,
            "evidence_repeats": evidence_repeats,
            "sensor_noise": sensor_noise,
            "hidden_dim": hidden_dim,
            "max_probes": max_probes,
            "unsafe_weight": unsafe_weight,
            "probe_weight": probe_weight,
            "seed": seed,
            "hidden_labels_used_as_features": False,
        },
        "diagnostics": {
            "train_contexts": len(train_contexts),
            "train_records": len(train_records),
        },
        "learners": {name: result.as_dict() for name, result in learners.items()},
    }


def generate_active_contexts(
    contexts_per_regime: int,
    sensor_noise: float,
    rng: np.random.Generator,
    sensor_base: SensorBase = ACTIVE_SENSOR_BASE,
) -> tuple[ActiveContext, ...]:
    contexts: list[ActiveContext] = []
    context_id = 0
    for regime in REGIMES:
        for _ in range(contexts_per_regime):
            sensor = np.asarray(sensor_base[regime], dtype=float)
            sensor += rng.normal(0.0, sensor_noise, size=len(sensor))
            contexts.append(
                ActiveContext(
                    context_id=context_id,
                    regime=regime,
                    sensor_features=tuple(float(value) for value in np.clip(sensor, 0.0, 1.0)),
                )
            )
            context_id += 1
    return tuple(contexts)


def generate_active_records(
    contexts: tuple[ActiveContext, ...],
    intervention_repeats: int,
    evidence_repeats: int,
    rng: np.random.Generator,
    probe_likelihood: ProbeLikelihood = PROBE_LIKELIHOOD,
    probe_unsafe: ProbeUnsafe = PROBE_UNSAFE,
) -> tuple[ActiveRecord, ...]:
    records: list[ActiveRecord] = []
    for context in contexts:
        evidence_states = [unknown_evidence()]
        for probe in PROBES:
            for _ in range(evidence_repeats):
                positive = bool(rng.binomial(1, probe_likelihood[probe][context.regime]))
                evidence_states.append(update_evidence(unknown_evidence(), probe, positive))
        for evidence in evidence_states:
            for test in TESTS:
                repeats = intervention_repeats if test in DIRECT_ACTIONS else max(1, intervention_repeats // 2)
                for _ in range(repeats):
                    records.append(
                        ActiveRecord(
                            context_id=context.context_id,
                            regime=context.regime,
                            sensor_features=context.sensor_features,
                            evidence_features=evidence,
                            test=test,
                            target=sample_target(context.regime, test, rng, probe_likelihood, probe_unsafe),
                        )
                    )
    return tuple(records)


def sample_target(
    regime: str,
    test: str,
    rng: np.random.Generator,
    probe_likelihood: ProbeLikelihood = PROBE_LIKELIHOOD,
    probe_unsafe: ProbeUnsafe = PROBE_UNSAFE,
) -> tuple[float, float]:
    if test in DIRECT_ACTIONS:
        outcome = ACTION_MODEL[regime][test]
        return (
            float(rng.binomial(1, outcome.success)),
            float(rng.binomial(1, outcome.unsafe)),
        )
    return (
        float(rng.binomial(1, probe_likelihood[test][regime])),
        float(rng.binomial(1, probe_unsafe[test])),
    )


def train_active_model(records: tuple[ActiveRecord, ...], hidden_dim: int, seed: int) -> TinyMLP:
    x = np.asarray([make_input(record.sensor_features, record.evidence_features, record.test) for record in records])
    y = np.asarray([record.target for record in records], dtype=float)
    model = TinyMLP.create(x.shape[1], hidden_dim, y.shape[1], seed)
    model.fit(x, y, epochs=800, learning_rate=0.025)
    return model


def make_input(sensor: tuple[float, ...], evidence: tuple[float, ...], test: str) -> np.ndarray:
    test_one_hot = np.asarray([1.0 if test == item else 0.0 for item in TESTS], dtype=float)
    return np.concatenate([np.asarray(sensor, dtype=float), np.asarray(evidence, dtype=float), test_one_hot])


def unknown_evidence() -> tuple[float, ...]:
    return tuple(UNKNOWN_EVIDENCE for _ in PROBES)


def update_evidence(evidence: tuple[float, ...], probe: str, positive: bool) -> tuple[float, ...]:
    values = list(evidence)
    values[PROBES.index(probe)] = 1.0 if positive else 0.0
    return tuple(values)


def predict(model: TinyMLP, sensor: tuple[float, ...], evidence: tuple[float, ...], test: str) -> np.ndarray:
    return model.predict(make_input(sensor, evidence, test)[None, :])[0]


def evaluate_policy(
    name: str,
    contexts: tuple[ActiveContext, ...],
    model: TinyMLP,
    max_probes: int,
    unsafe_weight: float,
    probe_weight: float,
    entropy_only: bool = False,
    probe_likelihood: ProbeLikelihood = PROBE_LIKELIHOOD,
    probe_unsafe: ProbeUnsafe = PROBE_UNSAFE,
) -> ActiveResult:
    totals = {"success": 0.0, "unsafe": 0.0, "probes": 0.0}
    by_regime: dict[str, dict[str, float | str]] = {}
    for regime in REGIMES:
        selected = [context for context in contexts if context.regime == regime]
        regime_totals = {"success": 0.0, "unsafe": 0.0, "probes": 0.0}
        action_counts: dict[str, float] = {}
        for context in selected:
            metrics = eval_tree(
                context=context,
                model=model,
                evidence=unknown_evidence(),
                used=(),
                path_probability=1.0,
                safe_probe_probability=1.0,
                n_probes=0,
                max_probes=max_probes,
                unsafe_weight=unsafe_weight,
                probe_weight=probe_weight,
                entropy_only=entropy_only,
                probe_likelihood=probe_likelihood,
                probe_unsafe=probe_unsafe,
            )
            for key in regime_totals:
                regime_totals[key] += float(metrics[key])
            action = str(metrics["dominant_action"])
            action_counts[action] = action_counts.get(action, 0.0) + float(metrics["dominant_weight"])
        n_regime = float(len(selected))
        for key in totals:
            totals[key] += regime_totals[key]
        by_regime[regime] = {
            "dominant_action": max(action_counts, key=action_counts.get),
            "success_rate": regime_totals["success"] / n_regime,
            "unsafe_failure_rate": regime_totals["unsafe"] / n_regime,
            "mean_probes": regime_totals["probes"] / n_regime,
        }
    n_total = float(len(contexts))
    success = totals["success"] / n_total
    unsafe = totals["unsafe"] / n_total
    probes = totals["probes"] / n_total
    return ActiveResult(
        name=name,
        success_rate=success,
        unsafe_failure_rate=unsafe,
        mean_probes=probes,
        risk_adjusted_score=success - unsafe_weight * unsafe - probe_weight * probes,
        by_regime=by_regime,
    )


def eval_tree(
    context: ActiveContext,
    model: TinyMLP,
    evidence: tuple[float, ...],
    used: tuple[str, ...],
    path_probability: float,
    safe_probe_probability: float,
    n_probes: int,
    max_probes: int,
    unsafe_weight: float,
    probe_weight: float,
    entropy_only: bool,
    probe_likelihood: ProbeLikelihood,
    probe_unsafe: ProbeUnsafe,
) -> dict[str, float | str]:
    decision = choose_decision(
        model=model,
        sensor=context.sensor_features,
        evidence=evidence,
        used=used,
        n_probes=n_probes,
        max_probes=max_probes,
        unsafe_weight=unsafe_weight,
        probe_weight=probe_weight,
        entropy_only=entropy_only,
    )
    if decision["kind"] == "act":
        action = str(decision["choice"])
        outcome = ACTION_MODEL[context.regime][action]
        return {
            "success": path_probability * safe_probe_probability * outcome.success,
            "unsafe": path_probability * (1.0 - safe_probe_probability * (1.0 - outcome.unsafe)),
            "probes": path_probability * n_probes,
            "dominant_action": action,
            "dominant_weight": path_probability,
        }

    probe = str(decision["choice"])
    parts = []
    for positive in (False, True):
        p_e = probe_likelihood[probe][context.regime] if positive else 1.0 - probe_likelihood[probe][context.regime]
        parts.append(
            eval_tree(
                context=context,
                model=model,
                evidence=update_evidence(evidence, probe, positive),
                used=used + (probe,),
                path_probability=path_probability * p_e,
                safe_probe_probability=safe_probe_probability * (1.0 - probe_unsafe[probe]),
                n_probes=n_probes + 1,
                max_probes=max_probes,
                unsafe_weight=unsafe_weight,
                probe_weight=probe_weight,
                entropy_only=entropy_only,
                probe_likelihood=probe_likelihood,
                probe_unsafe=probe_unsafe,
            )
        )
    action_weights: dict[str, float] = {}
    for part in parts:
        action = str(part["dominant_action"])
        action_weights[action] = action_weights.get(action, 0.0) + float(part["dominant_weight"])
    return {
        "success": sum(float(part["success"]) for part in parts),
        "unsafe": sum(float(part["unsafe"]) for part in parts),
        "probes": sum(float(part["probes"]) for part in parts),
        "dominant_action": max(action_weights, key=action_weights.get),
        "dominant_weight": sum(action_weights.values()),
    }


def choose_decision(
    model: TinyMLP,
    sensor: tuple[float, ...],
    evidence: tuple[float, ...],
    used: tuple[str, ...],
    n_probes: int,
    max_probes: int,
    unsafe_weight: float,
    probe_weight: float,
    entropy_only: bool,
) -> dict[str, float | str]:
    direct = direct_decision(model, sensor, evidence, unsafe_weight)
    if n_probes >= max_probes:
        return direct

    candidates = [direct]
    for probe in PROBES:
        if probe in used:
            continue
        pred = predict(model, sensor, evidence, probe)
        p_pos = float(np.clip(pred[0], 0.0, 1.0))
        p_unsafe = float(np.clip(pred[1], 0.0, 1.0))
        pos_score = model_value(
            model,
            sensor,
            update_evidence(evidence, probe, True),
            used + (probe,),
            n_probes + 1,
            max_probes,
            unsafe_weight,
            probe_weight,
            entropy_only,
        )
        neg_score = model_value(
            model,
            sensor,
            update_evidence(evidence, probe, False),
            used + (probe,),
            n_probes + 1,
            max_probes,
            unsafe_weight,
            probe_weight,
            entropy_only,
        )
        if entropy_only:
            score = binary_entropy(p_pos)
        else:
            score = (1.0 - p_unsafe) * (p_pos * pos_score + (1.0 - p_pos) * neg_score)
            score -= unsafe_weight * p_unsafe + probe_weight
        candidates.append({"kind": "probe", "choice": probe, "score": score})
    candidates.sort(key=lambda item: (-float(item["score"]), str(item["choice"])))
    return candidates[0]


def model_value(
    model: TinyMLP,
    sensor: tuple[float, ...],
    evidence: tuple[float, ...],
    used: tuple[str, ...],
    n_probes: int,
    max_probes: int,
    unsafe_weight: float,
    probe_weight: float,
    entropy_only: bool,
) -> float:
    decision = choose_decision(
        model,
        sensor,
        evidence,
        used,
        n_probes,
        max_probes,
        unsafe_weight,
        probe_weight,
        entropy_only,
    )
    if decision["kind"] == "act":
        return float(decision["score"])
    probe = str(decision["choice"])
    pred = predict(model, sensor, evidence, probe)
    p_pos = float(np.clip(pred[0], 0.0, 1.0))
    p_unsafe = float(np.clip(pred[1], 0.0, 1.0))
    pos_score = model_value(
        model,
        sensor,
        update_evidence(evidence, probe, True),
        used + (probe,),
        n_probes + 1,
        max_probes,
        unsafe_weight,
        probe_weight,
        entropy_only,
    )
    neg_score = model_value(
        model,
        sensor,
        update_evidence(evidence, probe, False),
        used + (probe,),
        n_probes + 1,
        max_probes,
        unsafe_weight,
        probe_weight,
        entropy_only,
    )
    return (1.0 - p_unsafe) * (p_pos * pos_score + (1.0 - p_pos) * neg_score) - unsafe_weight * p_unsafe - probe_weight


def direct_decision(
    model: TinyMLP,
    sensor: tuple[float, ...],
    evidence: tuple[float, ...],
    unsafe_weight: float,
) -> dict[str, float | str]:
    candidates = []
    for action in DIRECT_ACTIONS:
        pred = predict(model, sensor, evidence, action)
        score = float(pred[0] - unsafe_weight * pred[1])
        candidates.append((score, action))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    score, action = candidates[0]
    return {"kind": "act", "choice": action, "score": score}


def binary_entropy(p: float) -> float:
    p = float(np.clip(p, 1e-8, 1.0 - 1e-8))
    return float(-(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p)))


def evaluate_oracle(
    contexts: tuple[ActiveContext, ...],
    unsafe_weight: float,
    probe_weight: float,
) -> ActiveResult:
    by_regime: dict[str, dict[str, float | str]] = {}
    total_success = 0.0
    total_unsafe = 0.0
    for regime in REGIMES:
        utilities = {
            action: ACTION_MODEL[regime][action].success - unsafe_weight * ACTION_MODEL[regime][action].unsafe
            for action in DIRECT_ACTIONS
        }
        action = max(utilities, key=utilities.get)
        outcome = ACTION_MODEL[regime][action]
        selected = [context for context in contexts if context.regime == regime]
        total_success += len(selected) * outcome.success
        total_unsafe += len(selected) * outcome.unsafe
        by_regime[regime] = {
            "dominant_action": action,
            "success_rate": outcome.success,
            "unsafe_failure_rate": outcome.unsafe,
            "mean_probes": 0.0,
        }
    n_total = float(len(contexts))
    success = total_success / n_total
    unsafe = total_unsafe / n_total
    return ActiveResult(
        name="oracle_regime",
        success_rate=success,
        unsafe_failure_rate=unsafe,
        mean_probes=0.0,
        risk_adjusted_score=success - unsafe_weight * unsafe - probe_weight * 0.0,
        by_regime=by_regime,
    )


def scaled_probe_likelihood(strength: float) -> ProbeLikelihood:
    """Move probe likelihoods toward 0.5 while preserving their sign."""
    return {
        probe: {
            regime: float(np.clip(0.5 + strength * (PROBE_LIKELIHOOD[probe][regime] - 0.5), 0.0, 1.0))
            for regime in REGIMES
        }
        for probe in PROBES
    }


def run_neural_active_boundary_benchmark(
    contexts_per_regime: int = 80,
    intervention_repeats: int = 4,
    evidence_repeats: int = 2,
    sensor_noise: float = 0.020,
    hidden_dim: int = 32,
    max_probes: int = 2,
    unsafe_weight: float = 2.0,
    seed: int = 71,
) -> dict[str, Any]:
    specs = {
        "ambiguous_informative": BoundarySpec(
            description="Initial sensors alias regime pairs and probes keep their default informativeness.",
            sensor_base=ACTIVE_SENSOR_BASE,
            probe_likelihood=PROBE_LIKELIHOOD,
            probe_weight=0.02,
        ),
        "distinct_informative": BoundarySpec(
            description="Initial sensors separate the regimes, so probing should have less marginal value.",
            sensor_base=DISTINCT_SENSOR_BASE,
            probe_likelihood=PROBE_LIKELIHOOD,
            probe_weight=0.02,
        ),
        "ambiguous_weak_probe": BoundarySpec(
            description="Initial sensors alias regime pairs but probes are weakly informative.",
            sensor_base=ACTIVE_SENSOR_BASE,
            probe_likelihood=scaled_probe_likelihood(0.25),
            probe_weight=0.02,
        ),
        "ambiguous_costly_probe": BoundarySpec(
            description="Initial sensors alias regime pairs but probes are costly.",
            sensor_base=ACTIVE_SENSOR_BASE,
            probe_likelihood=PROBE_LIKELIHOOD,
            probe_weight=0.12,
        ),
    }
    cases: dict[str, Any] = {}
    for index, (name, spec) in enumerate(specs.items()):
        result = run_neural_active_probe_benchmark(
            contexts_per_regime=contexts_per_regime,
            intervention_repeats=intervention_repeats,
            evidence_repeats=evidence_repeats,
            sensor_noise=sensor_noise,
            hidden_dim=hidden_dim,
            max_probes=max_probes,
            unsafe_weight=unsafe_weight,
            probe_weight=spec.probe_weight,
            seed=seed + 17 * index,
            sensor_base=spec.sensor_base,
            probe_likelihood=spec.probe_likelihood,
        )
        no_probe = result["learners"]["learned_no_probe"]
        active = result["learners"]["learned_active_probe"]
        entropy = result["learners"]["learned_entropy_probe"]
        cases[name] = {
            "description": spec.description,
            "learners": result["learners"],
            "summary": {
                "active_minus_no_probe_score": active["risk_adjusted_score"] - no_probe["risk_adjusted_score"],
                "active_minus_entropy_score": active["risk_adjusted_score"] - entropy["risk_adjusted_score"],
                "no_probe_minus_active_unsafe": no_probe["unsafe_failure_rate"] - active["unsafe_failure_rate"],
                "active_mean_probes": active["mean_probes"],
            },
        }
    return {
        "benchmark": "neural_active_boundary",
        "description": (
            "Boundary-condition sweep for the learned active-probing claim. "
            "It tests whether gains are largest when sensors are aliased and probes are informative, "
            "and whether the claim weakens when sensors already identify the regime or probes are weak/costly."
        ),
        "config": {
            "contexts_per_regime": contexts_per_regime,
            "intervention_repeats": intervention_repeats,
            "evidence_repeats": evidence_repeats,
            "sensor_noise": sensor_noise,
            "hidden_dim": hidden_dim,
            "max_probes": max_probes,
            "unsafe_weight": unsafe_weight,
            "seed": seed,
            "hidden_labels_used_as_features": False,
        },
        "cases": cases,
        "summary": {
            "ambiguous_margin": cases["ambiguous_informative"]["summary"]["active_minus_no_probe_score"],
            "distinct_margin": cases["distinct_informative"]["summary"]["active_minus_no_probe_score"],
            "weak_probe_margin": cases["ambiguous_weak_probe"]["summary"]["active_minus_no_probe_score"],
            "costly_probe_margin": cases["ambiguous_costly_probe"]["summary"]["active_minus_no_probe_score"],
            "ambiguous_active_minus_entropy": cases["ambiguous_informative"]["summary"][
                "active_minus_entropy_score"
            ],
            "costly_active_mean_probes": cases["ambiguous_costly_probe"]["summary"]["active_mean_probes"],
        },
    }


def run_neural_active_seed_sweep_benchmark(
    seeds: tuple[int, ...] = (53, 59, 61, 67, 71),
    contexts_per_regime: int = 80,
    intervention_repeats: int = 4,
    evidence_repeats: int = 2,
    sensor_noise: float = 0.020,
    hidden_dim: int = 32,
    max_probes: int = 2,
    unsafe_weight: float = 2.0,
    probe_weight: float = 0.02,
) -> dict[str, Any]:
    cases: dict[str, Any] = {}
    active_minus_no_probe: list[float] = []
    no_probe_minus_active_unsafe: list[float] = []
    active_minus_entropy: list[float] = []
    oracle_gaps: list[float] = []
    active_scores: list[float] = []
    for seed in seeds:
        result = run_neural_active_probe_benchmark(
            contexts_per_regime=contexts_per_regime,
            intervention_repeats=intervention_repeats,
            evidence_repeats=evidence_repeats,
            sensor_noise=sensor_noise,
            hidden_dim=hidden_dim,
            max_probes=max_probes,
            unsafe_weight=unsafe_weight,
            probe_weight=probe_weight,
            seed=seed,
        )
        no_probe = result["learners"]["learned_no_probe"]
        active = result["learners"]["learned_active_probe"]
        entropy = result["learners"]["learned_entropy_probe"]
        oracle = result["learners"]["oracle_regime"]
        score_margin = active["risk_adjusted_score"] - no_probe["risk_adjusted_score"]
        unsafe_margin = no_probe["unsafe_failure_rate"] - active["unsafe_failure_rate"]
        entropy_margin = active["risk_adjusted_score"] - entropy["risk_adjusted_score"]
        oracle_gap = oracle["risk_adjusted_score"] - active["risk_adjusted_score"]
        active_score = active["risk_adjusted_score"]
        active_minus_no_probe.append(score_margin)
        no_probe_minus_active_unsafe.append(unsafe_margin)
        active_minus_entropy.append(entropy_margin)
        oracle_gaps.append(oracle_gap)
        active_scores.append(active_score)
        cases[str(seed)] = {
            "learners": result["learners"],
            "summary": {
                "active_minus_no_probe_score": score_margin,
                "no_probe_minus_active_unsafe": unsafe_margin,
                "active_minus_entropy_score": entropy_margin,
                "oracle_gap": oracle_gap,
                "active_score": active_score,
                "active_mean_probes": active["mean_probes"],
            },
        }
    return {
        "benchmark": "neural_active_seed_sweep",
        "description": (
            "Multi-seed robustness sweep for the learned active-probing benchmark. "
            "It repeats the aliased-sensor, informative-probe setting over deterministic seeds."
        ),
        "config": {
            "seeds": list(seeds),
            "contexts_per_regime": contexts_per_regime,
            "intervention_repeats": intervention_repeats,
            "evidence_repeats": evidence_repeats,
            "sensor_noise": sensor_noise,
            "hidden_dim": hidden_dim,
            "max_probes": max_probes,
            "unsafe_weight": unsafe_weight,
            "probe_weight": probe_weight,
            "hidden_labels_used_as_features": False,
        },
        "cases": cases,
        "summary": {
            "active_score_mean": float(np.mean(active_scores)),
            "active_score_std": float(np.std(active_scores)),
            "active_minus_no_probe_mean": float(np.mean(active_minus_no_probe)),
            "active_minus_no_probe_min": float(np.min(active_minus_no_probe)),
            "no_probe_minus_active_unsafe_mean": float(np.mean(no_probe_minus_active_unsafe)),
            "no_probe_minus_active_unsafe_min": float(np.min(no_probe_minus_active_unsafe)),
            "active_minus_entropy_mean": float(np.mean(active_minus_entropy)),
            "active_minus_entropy_min": float(np.min(active_minus_entropy)),
            "oracle_gap_mean": float(np.mean(oracle_gaps)),
            "oracle_gap_max": float(np.max(oracle_gaps)),
        },
    }
