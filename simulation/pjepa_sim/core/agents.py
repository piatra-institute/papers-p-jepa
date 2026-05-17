"""Policy evaluation for the hidden-regime P-JEPA simulation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pjepa_sim.core.dishworld import (
    ACTION_MODEL,
    DIRECT_ACTIONS,
    PRIOR,
    PROBE_LIKELIHOOD,
    PROBE_UNSAFE,
    PROBES,
    REGIMES,
    bayes_update,
    best_direct_action,
    evidence_probability,
    expected_action_outcome,
    obstruction,
    prediction_matrix,
)


@dataclass
class PolicyResult:
    name: str
    success_rate: float = 0.0
    unsafe_failure_rate: float = 0.0
    mean_probes: float = 0.0
    mean_obstruction_start: float = 0.0
    mean_obstruction_at_action: float = 0.0
    mean_obstruction_reduction: float = 0.0
    by_regime: dict[str, dict[str, float | str]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "success_rate": self.success_rate,
            "unsafe_failure_rate": self.unsafe_failure_rate,
            "mean_probes": self.mean_probes,
            "mean_obstruction_start": self.mean_obstruction_start,
            "mean_obstruction_at_action": self.mean_obstruction_at_action,
            "mean_obstruction_reduction": self.mean_obstruction_reduction,
            "by_regime": self.by_regime,
        }


def _action_result(name: str, action: str, posterior: np.ndarray | None = None) -> PolicyResult:
    """Evaluate a no-probe policy under a uniform test distribution."""
    posterior = PRIOR if posterior is None else posterior
    start_obstruction = obstruction(PRIOR)
    result = PolicyResult(
        name=name,
        mean_obstruction_start=start_obstruction,
        mean_obstruction_at_action=start_obstruction,
        mean_obstruction_reduction=0.0,
    )
    for regime in REGIMES:
        outcome = ACTION_MODEL[regime][action]
        result.by_regime[regime] = {
            "action": action,
            "success_rate": outcome.success,
            "unsafe_failure_rate": outcome.unsafe,
            "mean_probes": 0.0,
            "obstruction_at_action": start_obstruction,
        }
    _aggregate(result)
    return result


def visual_policy() -> PolicyResult:
    """Appearance-only baseline: every visible plate is treated as dry."""
    return _action_result("visual_policy", "lift_fast")


def mixture_no_glue() -> PolicyResult:
    """Local dry/soapy/cracked/heavy models exist, but visible gating selects dry."""
    return _action_result("mixture_no_glue", "lift_fast")


def psr_only() -> PolicyResult:
    """Action-conditioned baseline that chooses the prior-best direct action."""
    action = best_direct_action(PRIOR)
    return _action_result("psr_only", action)


def jepa_prior() -> PolicyResult:
    """Latent-prediction baseline collapsed to the prior mean representation."""
    action = best_direct_action(PRIOR)
    return _action_result("jepa_prior", action)


def model_based_prior() -> PolicyResult:
    """Ordinary transition-model baseline using the prior outcome table."""
    action = best_direct_action(PRIOR)
    return _action_result("model_based_prior", action)


def expected_probe_reduction(posterior: np.ndarray, probe: str) -> float:
    """Expected decrease in obstruction after one probe."""
    before = obstruction(posterior)
    after = 0.0
    for positive in (False, True):
        p_e = evidence_probability(posterior, probe, positive)
        post_e = bayes_update(posterior, probe, positive)
        after += p_e * obstruction(post_e)
    return before - after


def choose_probe(posterior: np.ndarray, used: tuple[str, ...]) -> str | None:
    remaining = [p for p in PROBES if p not in used]
    if not remaining:
        return None
    scores = [(expected_probe_reduction(posterior, p), p) for p in remaining]
    scores.sort(key=lambda item: (-item[0], item[1]))
    return scores[0][1]


def sheaf_probe(
    threshold: float = 0.060,
    max_probes: int = 3,
) -> PolicyResult:
    """Evaluate obstruction-driven probing by exact evidence enumeration."""
    start_obstruction = obstruction(PRIOR)
    result = PolicyResult(name="sheaf_probe", mean_obstruction_start=start_obstruction)
    for regime in REGIMES:
        regime_index = REGIMES.index(regime)
        metrics = _eval_probe_tree(
            true_regime=regime,
            posterior=PRIOR,
            used=(),
            path_probability=1.0,
            safe_probe_probability=1.0,
            threshold=threshold,
            max_probes=max_probes,
            n_probes=0,
        )
        result.by_regime[regime] = {
            "action": metrics["dominant_action"],
            "success_rate": metrics["success"],
            "unsafe_failure_rate": metrics["unsafe"],
            "mean_probes": metrics["probes"],
            "obstruction_at_action": metrics["obstruction_at_action"],
        }
    _aggregate(result)
    return result


def _eval_probe_tree(
    true_regime: str,
    posterior: np.ndarray,
    used: tuple[str, ...],
    path_probability: float,
    safe_probe_probability: float,
    threshold: float,
    max_probes: int,
    n_probes: int,
) -> dict[str, float | str]:
    current_obstruction = obstruction(posterior)
    if current_obstruction <= threshold or n_probes >= max_probes:
        action = best_direct_action(posterior)
        direct = ACTION_MODEL[true_regime][action]
        return {
            "success": path_probability * safe_probe_probability * direct.success,
            "unsafe": path_probability * (1.0 - safe_probe_probability * (1.0 - direct.unsafe)),
            "probes": path_probability * n_probes,
            "obstruction_at_action": path_probability * current_obstruction,
            "dominant_action": action,
            "dominant_weight": path_probability,
        }

    probe = choose_probe(posterior, used)
    if probe is None:
        action = best_direct_action(posterior)
        direct = ACTION_MODEL[true_regime][action]
        return {
            "success": path_probability * safe_probe_probability * direct.success,
            "unsafe": path_probability * (1.0 - safe_probe_probability * (1.0 - direct.unsafe)),
            "probes": path_probability * n_probes,
            "obstruction_at_action": path_probability * current_obstruction,
            "dominant_action": action,
            "dominant_weight": path_probability,
        }

    parts = []
    for positive in (False, True):
        p_e = PROBE_LIKELIHOOD[probe][true_regime]
        if not positive:
            p_e = 1.0 - p_e
        updated = bayes_update(posterior, probe, positive)
        parts.append(
            _eval_probe_tree(
                true_regime=true_regime,
                posterior=updated,
                used=used + (probe,),
                path_probability=path_probability * p_e,
                safe_probe_probability=safe_probe_probability * (1.0 - PROBE_UNSAFE[probe]),
                threshold=threshold,
                max_probes=max_probes,
                n_probes=n_probes + 1,
            )
        )

    success = sum(float(p["success"]) for p in parts)
    unsafe = sum(float(p["unsafe"]) for p in parts)
    probes = sum(float(p["probes"]) for p in parts)
    obstruction_at_action = sum(float(p["obstruction_at_action"]) for p in parts)
    action_weights: dict[str, float] = {}
    for p in parts:
        action = str(p["dominant_action"])
        action_weights[action] = action_weights.get(action, 0.0) + float(p["dominant_weight"])
    dominant_action = max(action_weights, key=action_weights.get)
    return {
        "success": success,
        "unsafe": unsafe,
        "probes": probes,
        "obstruction_at_action": obstruction_at_action,
        "dominant_action": dominant_action,
        "dominant_weight": sum(action_weights.values()),
    }


def representative_trace(true_regime: str = "soapy", threshold: float = 0.060) -> list[dict]:
    """Most-likely evidence path for a single hidden regime."""
    posterior = PRIOR.copy()
    trace = [
        {
            "step": "observe",
            "posterior": _posterior_dict(posterior),
            "obstruction": obstruction(posterior),
        }
    ]
    used: tuple[str, ...] = ()
    while obstruction(posterior) > threshold and len(used) < len(PROBES):
        probe = choose_probe(posterior, used)
        assert probe is not None
        positive = PROBE_LIKELIHOOD[probe][true_regime] >= 0.5
        posterior = bayes_update(posterior, probe, positive)
        used = used + (probe,)
        trace.append(
            {
                "step": probe,
                "evidence": "positive" if positive else "negative",
                "posterior": _posterior_dict(posterior),
                "obstruction": obstruction(posterior),
            }
        )
    action = best_direct_action(posterior)
    outcome = expected_action_outcome(np.eye(len(REGIMES))[REGIMES.index(true_regime)], action)
    trace.append(
        {
            "step": "act",
            "action": action,
            "success_probability": outcome.success,
            "unsafe_probability": outcome.unsafe,
            "posterior": _posterior_dict(posterior),
            "obstruction": obstruction(posterior),
        }
    )
    return trace


def _posterior_dict(posterior: np.ndarray) -> dict[str, float]:
    return {r: float(v) for r, v in zip(REGIMES, posterior)}


def _aggregate(result: PolicyResult) -> None:
    n = len(REGIMES)
    result.success_rate = float(sum(float(v["success_rate"]) for v in result.by_regime.values()) / n)
    result.unsafe_failure_rate = float(sum(float(v["unsafe_failure_rate"]) for v in result.by_regime.values()) / n)
    result.mean_probes = float(sum(float(v["mean_probes"]) for v in result.by_regime.values()) / n)
    result.mean_obstruction_at_action = float(
        sum(float(v["obstruction_at_action"]) for v in result.by_regime.values()) / n
    )
    result.mean_obstruction_start = obstruction(PRIOR)
    result.mean_obstruction_reduction = result.mean_obstruction_start - result.mean_obstruction_at_action


def agent_results() -> list[PolicyResult]:
    return [
        visual_policy(),
        model_based_prior(),
        jepa_prior(),
        psr_only(),
        mixture_no_glue(),
        sheaf_probe(),
    ]


def regime_prediction_table() -> dict[str, dict[str, float]]:
    preds = prediction_matrix()
    return {
        regime: {action: float(preds[i, j]) for j, action in enumerate(DIRECT_ACTIONS)}
        for i, regime in enumerate(REGIMES)
    }
