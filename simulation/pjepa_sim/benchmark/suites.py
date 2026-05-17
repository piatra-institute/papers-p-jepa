"""Benchmark suite definitions and exact evaluators for P-JEPA."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pjepa_sim.core.dishworld import (
    ACTION_MODEL,
    DIRECT_ACTIONS,
    PRIOR,
    PROBE_LIKELIHOOD,
    PROBE_UNSAFE,
    PROBES,
    REGIMES,
)
from pjepa_sim.paths import CONFIG_DIR


SUITE_ORDER = [
    "hidden_regime_v0",
    "heldout_regime_v0",
    "noisy_probe_v0",
    "costly_probe_v0",
    "miscalibrated_sections_v0",
]


@dataclass(frozen=True)
class ActionOutcome:
    success: float
    unsafe: float


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    description: str
    regimes: tuple[str, ...]
    direct_actions: tuple[str, ...]
    probes: tuple[str, ...]
    policy_prior: np.ndarray
    eval_prior: np.ndarray
    action_model: dict[str, dict[str, ActionOutcome]]
    belief_action_model: dict[str, dict[str, ActionOutcome]]
    probe_likelihood: dict[str, dict[str, float]]
    belief_probe_likelihood: dict[str, dict[str, float]]
    probe_unsafe: dict[str, float]
    visual_action: str
    sheaf_threshold: float = 0.060
    max_probes: int = 3
    unsafe_weight: float = 2.0
    probe_weight: float = 0.02

    def prior_dict(self, prior: np.ndarray) -> dict[str, float]:
        return {r: float(v) for r, v in zip(self.regimes, prior)}


@dataclass
class AgentScore:
    name: str
    success_rate: float = 0.0
    unsafe_failure_rate: float = 0.0
    mean_probes: float = 0.0
    mean_obstruction_start: float = 0.0
    mean_obstruction_at_action: float = 0.0
    mean_obstruction_reduction: float = 0.0
    risk_adjusted_score: float = 0.0
    by_regime: dict[str, dict[str, float | str]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "success_rate": self.success_rate,
            "unsafe_failure_rate": self.unsafe_failure_rate,
            "mean_probes": self.mean_probes,
            "mean_obstruction_start": self.mean_obstruction_start,
            "mean_obstruction_at_action": self.mean_obstruction_at_action,
            "mean_obstruction_reduction": self.mean_obstruction_reduction,
            "risk_adjusted_score": self.risk_adjusted_score,
            "by_regime": self.by_regime,
        }


def available_suites() -> list[str]:
    discovered = {path.stem for path in CONFIG_DIR.glob("*.json")}
    ordered = [name for name in SUITE_ORDER if name in discovered]
    extra = sorted(discovered - set(ordered))
    return ordered + extra


def load_spec(name: str) -> BenchmarkSpec:
    path = CONFIG_DIR / f"{name}.json"
    if not path.exists():
        known = ", ".join(available_suites())
        raise ValueError(f"unknown suite {name!r}; known suites: {known}")
    with path.open() as f:
        config = json.load(f)
    return spec_from_config(config)


def spec_from_config(config: dict[str, Any]) -> BenchmarkSpec:
    regimes = tuple(config.get("regimes", REGIMES))
    direct_actions = tuple(config.get("direct_actions", DIRECT_ACTIONS))
    probes = tuple(config.get("probes", PROBES))
    action_model = _merge_action_model(config.get("action_model", {}), regimes, direct_actions)
    belief_action_model = _merge_action_model(
        config.get("belief_action_model", config.get("action_model", {})),
        regimes,
        direct_actions,
    )
    probe_likelihood = _merge_probe_likelihood(config.get("probe_likelihood", {}), regimes, probes)
    belief_probe_likelihood = _merge_probe_likelihood(
        config.get("belief_probe_likelihood", config.get("probe_likelihood", {})),
        regimes,
        probes,
    )
    probe_unsafe = {probe: float(PROBE_UNSAFE[probe]) for probe in probes}
    for probe, value in config.get("probe_unsafe", {}).items():
        probe_unsafe[probe] = float(value)

    policy_prior = _prior_array(config.get("policy_prior"), regimes, PRIOR)
    eval_prior = _prior_array(config.get("eval_prior"), regimes, policy_prior)

    return BenchmarkSpec(
        name=str(config["name"]),
        description=str(config.get("description", "")),
        regimes=regimes,
        direct_actions=direct_actions,
        probes=probes,
        policy_prior=policy_prior,
        eval_prior=eval_prior,
        action_model=action_model,
        belief_action_model=belief_action_model,
        probe_likelihood=probe_likelihood,
        belief_probe_likelihood=belief_probe_likelihood,
        probe_unsafe=probe_unsafe,
        visual_action=str(config.get("visual_action", "lift_fast")),
        sheaf_threshold=float(config.get("sheaf_threshold", 0.060)),
        max_probes=int(config.get("max_probes", 3)),
        unsafe_weight=float(config.get("unsafe_weight", 2.0)),
        probe_weight=float(config.get("probe_weight", 0.02)),
    )


def evaluate_suite(spec: BenchmarkSpec, agent_names: list[str] | None = None) -> dict[str, Any]:
    all_agents = {
        "visual_policy": lambda: _fixed_action_result(spec, "visual_policy", spec.visual_action),
        "model_based_prior": lambda: _prior_action_result(spec, "model_based_prior"),
        "viability_prior": lambda: _utility_prior_result(spec, "viability_prior"),
        "jepa_prior": lambda: _prior_action_result(spec, "jepa_prior"),
        "psr_only": lambda: _prior_action_result(spec, "psr_only"),
        "active_psr_probe": lambda: _active_psr_probe_result(spec),
        "mixture_no_glue": lambda: _fixed_action_result(spec, "mixture_no_glue", spec.visual_action),
        "entropy_probe": lambda: _entropy_probe_result(spec),
        "sheaf_probe": lambda: _sheaf_probe_result(spec),
        "p_jepa_stack": lambda: _p_jepa_stack_result(spec),
        "oracle_hidden_regime": lambda: _oracle_result(spec),
    }
    selected = list(all_agents) if agent_names is None else agent_names
    unknown = [name for name in selected if name not in all_agents]
    if unknown:
        raise ValueError(f"unknown agent(s): {', '.join(unknown)}")

    agents = {name: all_agents[name]().as_dict() for name in selected}
    return {
        "suite": {
            "name": spec.name,
            "description": spec.description,
            "policy_prior": spec.prior_dict(spec.policy_prior),
            "eval_prior": spec.prior_dict(spec.eval_prior),
            "regimes": list(spec.regimes),
            "direct_actions": list(spec.direct_actions),
            "probes": list(spec.probes),
            "sheaf_threshold": spec.sheaf_threshold,
            "max_probes": spec.max_probes,
            "unsafe_weight": spec.unsafe_weight,
            "probe_weight": spec.probe_weight,
            "belief_model": {
                "action_model_differs_from_world": _action_models_differ(spec.action_model, spec.belief_action_model),
                "probe_likelihood_differs_from_world": spec.probe_likelihood != spec.belief_probe_likelihood,
            },
        },
        "obstruction": {
            "policy_prior": obstruction(spec, spec.policy_prior),
            "eval_prior": obstruction(spec, spec.eval_prior),
        },
        "agents": agents,
    }


def prediction_matrix(spec: BenchmarkSpec, *, belief: bool = True) -> np.ndarray:
    model = spec.belief_action_model if belief else spec.action_model
    return np.array(
        [
            [model[regime][action].success for action in spec.direct_actions]
            for regime in spec.regimes
        ],
        dtype=float,
    )


def obstruction(spec: BenchmarkSpec, posterior: np.ndarray) -> float:
    preds = prediction_matrix(spec)
    mean = posterior @ preds
    diffs = preds - mean
    return float(np.sum(posterior[:, None] * diffs * diffs))


def best_direct_action(spec: BenchmarkSpec, posterior: np.ndarray, *, belief: bool = True) -> str:
    expected_success = posterior @ prediction_matrix(spec, belief=belief)
    return spec.direct_actions[int(np.argmax(expected_success))]


def best_utility_action(spec: BenchmarkSpec, posterior: np.ndarray) -> str:
    utilities = [_expected_action_utility(spec, posterior, action) for action in spec.direct_actions]
    return spec.direct_actions[int(np.argmax(utilities))]


def _expected_action_utility(spec: BenchmarkSpec, posterior: np.ndarray, action: str) -> float:
    outcome = _expected_action_outcome(spec, posterior, action)
    return outcome.success - spec.unsafe_weight * outcome.unsafe


def _expected_action_outcome(spec: BenchmarkSpec, posterior: np.ndarray, action: str) -> ActionOutcome:
    success = 0.0
    unsafe = 0.0
    for i, regime in enumerate(spec.regimes):
        outcome = spec.belief_action_model[regime][action]
        success += posterior[i] * outcome.success
        unsafe += posterior[i] * outcome.unsafe
    return ActionOutcome(success=float(success), unsafe=float(unsafe))


def bayes_update(spec: BenchmarkSpec, posterior: np.ndarray, probe: str, positive: bool) -> np.ndarray:
    likelihood = np.array([spec.belief_probe_likelihood[probe][r] for r in spec.regimes], dtype=float)
    if not positive:
        likelihood = 1.0 - likelihood
    updated = posterior * likelihood
    total = float(updated.sum())
    if total <= 0:
        return posterior.copy()
    return updated / total


def evidence_probability(spec: BenchmarkSpec, posterior: np.ndarray, probe: str, positive: bool) -> float:
    likelihood = np.array([spec.belief_probe_likelihood[probe][r] for r in spec.regimes], dtype=float)
    if not positive:
        likelihood = 1.0 - likelihood
    return float(np.dot(posterior, likelihood))


def expected_probe_reduction(spec: BenchmarkSpec, posterior: np.ndarray, probe: str) -> float:
    before = obstruction(spec, posterior)
    after = 0.0
    for positive in (False, True):
        p_e = evidence_probability(spec, posterior, probe, positive)
        after += p_e * obstruction(spec, bayes_update(spec, posterior, probe, positive))
    return before - after


def posterior_entropy(posterior: np.ndarray) -> float:
    positive = posterior[posterior > 0.0]
    return float(-np.sum(positive * np.log2(positive)))


def expected_entropy_reduction(spec: BenchmarkSpec, posterior: np.ndarray, probe: str) -> float:
    before = posterior_entropy(posterior)
    after = 0.0
    for positive in (False, True):
        p_e = evidence_probability(spec, posterior, probe, positive)
        after += p_e * posterior_entropy(bayes_update(spec, posterior, probe, positive))
    return before - after


def choose_probe(spec: BenchmarkSpec, posterior: np.ndarray, used: tuple[str, ...]) -> str | None:
    remaining = [probe for probe in spec.probes if probe not in used]
    if not remaining:
        return None
    scores = [(expected_probe_reduction(spec, posterior, probe), probe) for probe in remaining]
    scores.sort(key=lambda item: (-item[0], item[1]))
    return scores[0][1]


def choose_entropy_probe(spec: BenchmarkSpec, posterior: np.ndarray, used: tuple[str, ...]) -> str | None:
    remaining = [probe for probe in spec.probes if probe not in used]
    if not remaining:
        return None
    scores = [(expected_entropy_reduction(spec, posterior, probe), probe) for probe in remaining]
    scores.sort(key=lambda item: (-item[0], item[1]))
    return scores[0][1]


def _fixed_action_result(spec: BenchmarkSpec, name: str, action: str) -> AgentScore:
    result = AgentScore(name=name, mean_obstruction_start=obstruction(spec, spec.policy_prior))
    for regime in spec.regimes:
        outcome = spec.action_model[regime][action]
        result.by_regime[regime] = {
            "action": action,
            "success_rate": outcome.success,
            "unsafe_failure_rate": outcome.unsafe,
            "mean_probes": 0.0,
            "obstruction_at_action": result.mean_obstruction_start,
        }
    _aggregate(spec, result)
    return result


def _prior_action_result(spec: BenchmarkSpec, name: str) -> AgentScore:
    return _fixed_action_result(spec, name, best_direct_action(spec, spec.policy_prior))


def _utility_prior_result(spec: BenchmarkSpec, name: str) -> AgentScore:
    return _fixed_action_result(spec, name, best_utility_action(spec, spec.policy_prior))


def _oracle_result(spec: BenchmarkSpec) -> AgentScore:
    result = AgentScore(name="oracle_hidden_regime", mean_obstruction_start=obstruction(spec, spec.policy_prior))
    for regime in spec.regimes:
        posterior = np.zeros(len(spec.regimes), dtype=float)
        posterior[spec.regimes.index(regime)] = 1.0
        action = best_direct_action(spec, posterior, belief=False)
        outcome = spec.action_model[regime][action]
        result.by_regime[regime] = {
            "action": action,
            "success_rate": outcome.success,
            "unsafe_failure_rate": outcome.unsafe,
            "mean_probes": 0.0,
            "obstruction_at_action": 0.0,
        }
    _aggregate(spec, result)
    return result


def _active_psr_probe_result(spec: BenchmarkSpec) -> AgentScore:
    return _decision_probe_result(spec, "active_psr_probe", use_obstruction_gate=False)


def _p_jepa_stack_result(spec: BenchmarkSpec) -> AgentScore:
    return _decision_probe_result(spec, "p_jepa_stack", use_obstruction_gate=True)


def _decision_probe_result(
    spec: BenchmarkSpec,
    name: str,
    use_obstruction_gate: bool,
) -> AgentScore:
    result = AgentScore(name=name, mean_obstruction_start=obstruction(spec, spec.policy_prior))
    for regime in spec.regimes:
        metrics = _eval_decision_tree(
            spec=spec,
            true_regime=regime,
            posterior=spec.policy_prior,
            used=(),
            path_probability=1.0,
            safe_probe_probability=1.0,
            n_probes=0,
            use_obstruction_gate=use_obstruction_gate,
        )
        result.by_regime[regime] = {
            "action": metrics["dominant_action"],
            "success_rate": metrics["success"],
            "unsafe_failure_rate": metrics["unsafe"],
            "mean_probes": metrics["probes"],
            "obstruction_at_action": metrics["obstruction_at_action"],
        }
    _aggregate(spec, result)
    return result


def _belief_decision_metrics(
    spec: BenchmarkSpec,
    posterior: np.ndarray,
    used: tuple[str, ...],
    n_probes: int,
    use_obstruction_gate: bool,
) -> dict[str, float | str]:
    direct = _belief_leaf_metrics(spec, posterior, n_probes=0)
    if n_probes >= spec.max_probes:
        return direct
    if use_obstruction_gate and obstruction(spec, posterior) <= spec.sheaf_threshold:
        return direct

    candidates = [direct]
    for probe in spec.probes:
        if probe in used:
            continue
        unsafe_probe = spec.probe_unsafe[probe]
        branch_parts = []
        for positive in (False, True):
            p_e = evidence_probability(spec, posterior, probe, positive)
            next_metrics = _belief_decision_metrics(
                spec=spec,
                posterior=bayes_update(spec, posterior, probe, positive),
                used=used + (probe,),
                n_probes=n_probes + 1,
                use_obstruction_gate=use_obstruction_gate,
            )
            branch_parts.append((p_e, next_metrics))

        success = (1.0 - unsafe_probe) * sum(p * float(m["success"]) for p, m in branch_parts)
        unsafe = unsafe_probe + (1.0 - unsafe_probe) * sum(p * float(m["unsafe"]) for p, m in branch_parts)
        probes = 1.0 + sum(p * float(m["probes"]) for p, m in branch_parts)
        obs_at_action = sum(p * float(m["obstruction_at_action"]) for p, m in branch_parts)
        candidates.append(
            {
                "kind": "probe",
                "choice": probe,
                "success": success,
                "unsafe": unsafe,
                "probes": probes,
                "obstruction_at_action": obs_at_action,
                "score": _score(spec, success, unsafe, probes),
                "dominant_action": f"probe:{probe}",
            }
        )

    candidates.sort(key=lambda item: (-float(item["score"]), str(item["choice"])))
    return candidates[0]


def _belief_leaf_metrics(spec: BenchmarkSpec, posterior: np.ndarray, n_probes: int) -> dict[str, float | str]:
    action = best_utility_action(spec, posterior)
    outcome = _expected_action_outcome(spec, posterior, action)
    return {
        "kind": "act",
        "choice": action,
        "success": outcome.success,
        "unsafe": outcome.unsafe,
        "probes": float(n_probes),
        "obstruction_at_action": obstruction(spec, posterior),
        "score": _score(spec, outcome.success, outcome.unsafe, float(n_probes)),
        "dominant_action": action,
    }


def _eval_decision_tree(
    spec: BenchmarkSpec,
    true_regime: str,
    posterior: np.ndarray,
    used: tuple[str, ...],
    path_probability: float,
    safe_probe_probability: float,
    n_probes: int,
    use_obstruction_gate: bool,
) -> dict[str, float | str]:
    decision = _belief_decision_metrics(spec, posterior, used, n_probes, use_obstruction_gate)
    if decision["kind"] == "act":
        return _utility_leaf_metrics(
            spec=spec,
            true_regime=true_regime,
            posterior=posterior,
            path_probability=path_probability,
            safe_probe_probability=safe_probe_probability,
            n_probes=n_probes,
        )

    probe = str(decision["choice"])
    parts = []
    for positive in (False, True):
        p_e = spec.probe_likelihood[probe][true_regime]
        if not positive:
            p_e = 1.0 - p_e
        parts.append(
            _eval_decision_tree(
                spec=spec,
                true_regime=true_regime,
                posterior=bayes_update(spec, posterior, probe, positive),
                used=used + (probe,),
                path_probability=path_probability * p_e,
                safe_probe_probability=safe_probe_probability * (1.0 - spec.probe_unsafe[probe]),
                n_probes=n_probes + 1,
                use_obstruction_gate=use_obstruction_gate,
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
        "obstruction_at_action": sum(float(part["obstruction_at_action"]) for part in parts),
        "dominant_action": max(action_weights, key=action_weights.get),
        "dominant_weight": sum(action_weights.values()),
    }


def _utility_leaf_metrics(
    spec: BenchmarkSpec,
    true_regime: str,
    posterior: np.ndarray,
    path_probability: float,
    safe_probe_probability: float,
    n_probes: int,
) -> dict[str, float | str]:
    action = best_utility_action(spec, posterior)
    outcome = spec.action_model[true_regime][action]
    return {
        "success": path_probability * safe_probe_probability * outcome.success,
        "unsafe": path_probability * (1.0 - safe_probe_probability * (1.0 - outcome.unsafe)),
        "probes": path_probability * n_probes,
        "obstruction_at_action": path_probability * obstruction(spec, posterior),
        "dominant_action": action,
        "dominant_weight": path_probability,
    }


def _sheaf_probe_result(spec: BenchmarkSpec) -> AgentScore:
    result = AgentScore(name="sheaf_probe", mean_obstruction_start=obstruction(spec, spec.policy_prior))
    for regime in spec.regimes:
        metrics = _eval_probe_tree(
            spec=spec,
            true_regime=regime,
            posterior=spec.policy_prior,
            used=(),
            path_probability=1.0,
            safe_probe_probability=1.0,
            n_probes=0,
        )
        result.by_regime[regime] = {
            "action": metrics["dominant_action"],
            "success_rate": metrics["success"],
            "unsafe_failure_rate": metrics["unsafe"],
            "mean_probes": metrics["probes"],
            "obstruction_at_action": metrics["obstruction_at_action"],
        }
    _aggregate(spec, result)
    return result


def _entropy_probe_result(spec: BenchmarkSpec) -> AgentScore:
    result = AgentScore(name="entropy_probe", mean_obstruction_start=obstruction(spec, spec.policy_prior))
    for regime in spec.regimes:
        metrics = _eval_probe_tree(
            spec=spec,
            true_regime=regime,
            posterior=spec.policy_prior,
            used=(),
            path_probability=1.0,
            safe_probe_probability=1.0,
            n_probes=0,
            chooser=choose_entropy_probe,
        )
        result.by_regime[regime] = {
            "action": metrics["dominant_action"],
            "success_rate": metrics["success"],
            "unsafe_failure_rate": metrics["unsafe"],
            "mean_probes": metrics["probes"],
            "obstruction_at_action": metrics["obstruction_at_action"],
        }
    _aggregate(spec, result)
    return result


def _eval_probe_tree(
    spec: BenchmarkSpec,
    true_regime: str,
    posterior: np.ndarray,
    used: tuple[str, ...],
    path_probability: float,
    safe_probe_probability: float,
    n_probes: int,
    chooser=choose_probe,
) -> dict[str, float | str]:
    current_obstruction = obstruction(spec, posterior)
    if current_obstruction <= spec.sheaf_threshold or n_probes >= spec.max_probes:
        return _leaf_metrics(spec, true_regime, posterior, path_probability, safe_probe_probability, n_probes)

    probe = chooser(spec, posterior, used)
    if probe is None:
        return _leaf_metrics(spec, true_regime, posterior, path_probability, safe_probe_probability, n_probes)

    parts = []
    for positive in (False, True):
        p_e = spec.probe_likelihood[probe][true_regime]
        if not positive:
            p_e = 1.0 - p_e
        parts.append(
            _eval_probe_tree(
                spec=spec,
                true_regime=true_regime,
                posterior=bayes_update(spec, posterior, probe, positive),
                used=used + (probe,),
                path_probability=path_probability * p_e,
                safe_probe_probability=safe_probe_probability * (1.0 - spec.probe_unsafe[probe]),
                n_probes=n_probes + 1,
                chooser=chooser,
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
        "obstruction_at_action": sum(float(part["obstruction_at_action"]) for part in parts),
        "dominant_action": max(action_weights, key=action_weights.get),
        "dominant_weight": sum(action_weights.values()),
    }


def _leaf_metrics(
    spec: BenchmarkSpec,
    true_regime: str,
    posterior: np.ndarray,
    path_probability: float,
    safe_probe_probability: float,
    n_probes: int,
) -> dict[str, float | str]:
    action = best_direct_action(spec, posterior)
    outcome = spec.action_model[true_regime][action]
    return {
        "success": path_probability * safe_probe_probability * outcome.success,
        "unsafe": path_probability * (1.0 - safe_probe_probability * (1.0 - outcome.unsafe)),
        "probes": path_probability * n_probes,
        "obstruction_at_action": path_probability * obstruction(spec, posterior),
        "dominant_action": action,
        "dominant_weight": path_probability,
    }


def _aggregate(spec: BenchmarkSpec, result: AgentScore) -> None:
    weights = {regime: float(weight) for regime, weight in zip(spec.regimes, spec.eval_prior)}
    result.success_rate = _weighted(result, weights, "success_rate")
    result.unsafe_failure_rate = _weighted(result, weights, "unsafe_failure_rate")
    result.mean_probes = _weighted(result, weights, "mean_probes")
    result.mean_obstruction_at_action = _weighted(result, weights, "obstruction_at_action")
    result.mean_obstruction_start = obstruction(spec, spec.policy_prior)
    result.mean_obstruction_reduction = result.mean_obstruction_start - result.mean_obstruction_at_action
    result.risk_adjusted_score = (
        _score(spec, result.success_rate, result.unsafe_failure_rate, result.mean_probes)
    )


def _score(spec: BenchmarkSpec, success: float, unsafe: float, probes: float) -> float:
    return success - spec.unsafe_weight * unsafe - spec.probe_weight * probes


def _weighted(result: AgentScore, weights: dict[str, float], key: str) -> float:
    return float(sum(weights[regime] * float(result.by_regime[regime][key]) for regime in weights))


def _merge_action_model(
    overrides: dict[str, Any],
    regimes: tuple[str, ...],
    direct_actions: tuple[str, ...],
) -> dict[str, dict[str, ActionOutcome]]:
    merged: dict[str, dict[str, ActionOutcome]] = {}
    for regime in regimes:
        merged[regime] = {}
        for action in direct_actions:
            base = ACTION_MODEL[regime][action]
            raw = overrides.get(regime, {}).get(action, {})
            merged[regime][action] = ActionOutcome(
                success=float(raw.get("success", base.success)),
                unsafe=float(raw.get("unsafe", base.unsafe)),
            )
    return merged


def _merge_probe_likelihood(
    overrides: dict[str, Any],
    regimes: tuple[str, ...],
    probes: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    merged: dict[str, dict[str, float]] = {}
    for probe in probes:
        merged[probe] = {}
        for regime in regimes:
            merged[probe][regime] = float(overrides.get(probe, {}).get(regime, PROBE_LIKELIHOOD[probe][regime]))
    return merged


def _prior_array(raw: dict[str, float] | None, regimes: tuple[str, ...], default: np.ndarray) -> np.ndarray:
    if raw is None:
        values = np.array([default[REGIMES.index(regime)] for regime in regimes], dtype=float)
    else:
        values = np.array([float(raw.get(regime, 0.0)) for regime in regimes], dtype=float)
    total = float(values.sum())
    if total <= 0:
        raise ValueError("prior must have positive mass")
    return values / total


def _action_models_differ(
    world: dict[str, dict[str, ActionOutcome]],
    belief: dict[str, dict[str, ActionOutcome]],
) -> bool:
    for regime, actions in world.items():
        for action, world_outcome in actions.items():
            belief_outcome = belief[regime][action]
            if (
                world_outcome.success != belief_outcome.success
                or world_outcome.unsafe != belief_outcome.unsafe
            ):
                return True
    return False
