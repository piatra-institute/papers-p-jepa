"""Pixel-observation continuous-control benchmark for P-JEPA.

This benchmark is deliberately local and lightweight. It moves one step beyond
structured sensor vectors by rendering small image observations and one step
beyond discrete dish actions by evaluating continuous 2D reach controllers
under hidden action dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from pjepa_sim.representation.neural import TinyMLP


REGIMES = ("nominal", "slippery", "fragile", "heavy")
CONTROLLERS = ("fast_direct", "slow_direct", "stabilized", "high_force")
PROBES = ("shear_probe", "tap_probe", "weigh_probe")
TESTS = CONTROLLERS + PROBES
UNKNOWN_EVIDENCE = 0.5
IMAGE_SIZE = 12

PROBE_LIKELIHOOD: dict[str, dict[str, float]] = {
    "shear_probe": {"nominal": 0.06, "slippery": 0.93, "fragile": 0.10, "heavy": 0.08},
    "tap_probe": {"nominal": 0.06, "slippery": 0.10, "fragile": 0.91, "heavy": 0.12},
    "weigh_probe": {"nominal": 0.08, "slippery": 0.08, "fragile": 0.12, "heavy": 0.92},
}
PROBE_UNSAFE = {"shear_probe": 0.01, "tap_probe": 0.02, "weigh_probe": 0.01}


@dataclass(frozen=True)
class Dynamics:
    scale: float
    noise: float
    unsafe_action_norm: float


DYNAMICS: dict[str, Dynamics] = {
    "nominal": Dynamics(scale=1.00, noise=0.020, unsafe_action_norm=1.60),
    "slippery": Dynamics(scale=1.35, noise=0.060, unsafe_action_norm=0.95),
    "fragile": Dynamics(scale=1.00, noise=0.015, unsafe_action_norm=0.60),
    "heavy": Dynamics(scale=0.38, noise=0.020, unsafe_action_norm=2.05),
}


@dataclass(frozen=True)
class PixelControlContext:
    context_id: int
    regime: str
    target: tuple[float, float]
    image_features: tuple[float, ...]


@dataclass(frozen=True)
class PixelControlRecord:
    context_id: int
    regime: str
    image_features: tuple[float, ...]
    evidence_features: tuple[float, ...]
    test: str
    target: tuple[float, float]


@dataclass(frozen=True)
class PixelControlResult:
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


def run_pixel_continuous_benchmark(
    contexts_per_regime: int = 64,
    intervention_repeats: int = 3,
    evidence_repeats: int = 3,
    hidden_dim: int = 56,
    max_probes: int = 2,
    unsafe_weight: float = 2.0,
    probe_weight: float = 0.01,
    seed: int = 83,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    train_contexts = generate_contexts(contexts_per_regime, rng)
    test_contexts = generate_contexts(contexts_per_regime, rng)
    train_records = generate_records(train_contexts, intervention_repeats, evidence_repeats, rng)
    model = train_model(train_records, hidden_dim, seed + 1)
    learners = {
        "pixel_no_probe": evaluate_policy(
            "pixel_no_probe",
            test_contexts,
            model,
            max_probes=0,
            unsafe_weight=unsafe_weight,
            probe_weight=probe_weight,
        ),
        "pixel_entropy_probe": evaluate_policy(
            "pixel_entropy_probe",
            test_contexts,
            model,
            max_probes=max_probes,
            unsafe_weight=unsafe_weight,
            probe_weight=probe_weight,
            entropy_only=True,
        ),
        "pixel_active_probe": evaluate_policy(
            "pixel_active_probe",
            test_contexts,
            model,
            max_probes=max_probes,
            unsafe_weight=unsafe_weight,
            probe_weight=probe_weight,
        ),
        "oracle_regime": evaluate_oracle(test_contexts, unsafe_weight, probe_weight),
    }
    return {
        "benchmark": "pixel_continuous_control",
        "description": (
            "Small rendered image observations alias pairs of hidden continuous-control regimes. "
            "A NumPy MLP predicts controller and probe outcomes from pixels, evidence, and test identities."
        ),
        "config": {
            "regimes": list(REGIMES),
            "controllers": list(CONTROLLERS),
            "probes": list(PROBES),
            "image_size": IMAGE_SIZE,
            "contexts_per_regime": contexts_per_regime,
            "intervention_repeats": intervention_repeats,
            "evidence_repeats": evidence_repeats,
            "hidden_dim": hidden_dim,
            "max_probes": max_probes,
            "unsafe_weight": unsafe_weight,
            "probe_weight": probe_weight,
            "seed": seed,
            "hidden_labels_used_as_features": False,
            "learner_inputs": "rendered_pixels + probe_evidence + test_one_hot",
        },
        "diagnostics": {
            "train_contexts": len(train_contexts),
            "train_records": len(train_records),
            "pixel_dim": len(train_contexts[0].image_features),
        },
        "learners": {name: result.as_dict() for name, result in learners.items()},
    }


def generate_contexts(contexts_per_regime: int, rng: np.random.Generator) -> tuple[PixelControlContext, ...]:
    contexts: list[PixelControlContext] = []
    context_id = 0
    for regime in REGIMES:
        for _ in range(contexts_per_regime):
            angle = float(rng.uniform(-np.pi, np.pi))
            radius = float(rng.uniform(0.92, 1.08))
            target = (radius * np.cos(angle), radius * np.sin(angle))
            image = render_context(regime, target, rng)
            contexts.append(
                PixelControlContext(
                    context_id=context_id,
                    regime=regime,
                    target=(float(target[0]), float(target[1])),
                    image_features=tuple(float(value) for value in image.reshape(-1)),
                )
            )
            context_id += 1
    return tuple(contexts)


def render_context(regime: str, target: tuple[float, float], rng: np.random.Generator) -> np.ndarray:
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=float)
    xx, yy = np.meshgrid(np.linspace(-1.2, 1.2, IMAGE_SIZE), np.linspace(-1.2, 1.2, IMAGE_SIZE))
    image += 0.12 * np.exp(-6.0 * (xx * xx + yy * yy))
    target_x, target_y = target
    image += 0.85 * np.exp(-35.0 * ((xx - target_x) ** 2 + (yy - target_y) ** 2))

    # The image intentionally aliases regime pairs: nominal/slippery and
    # fragile/heavy share texture, so pixels alone cannot select the safe
    # controller within each pair.
    if regime in ("fragile", "heavy"):
        image += 0.16 * (np.abs(xx - yy) < 0.18)
    else:
        image += 0.06 * (np.abs(xx + yy) < 0.22)

    image += rng.normal(0.0, 0.025, size=image.shape)
    image *= float(rng.uniform(0.92, 1.08))
    return np.clip(image, 0.0, 1.0)


def generate_records(
    contexts: tuple[PixelControlContext, ...],
    intervention_repeats: int,
    evidence_repeats: int,
    rng: np.random.Generator,
) -> tuple[PixelControlRecord, ...]:
    records: list[PixelControlRecord] = []
    for context in contexts:
        evidence_states = [unknown_evidence()]
        for probe in PROBES:
            for _ in range(evidence_repeats):
                positive = bool(rng.binomial(1, PROBE_LIKELIHOOD[probe][context.regime]))
                evidence_states.append(update_evidence(unknown_evidence(), probe, positive))
        for evidence in evidence_states:
            for test in TESTS:
                repeats = intervention_repeats if test in CONTROLLERS else max(1, intervention_repeats // 2)
                for _ in range(repeats):
                    records.append(
                        PixelControlRecord(
                            context_id=context.context_id,
                            regime=context.regime,
                            image_features=context.image_features,
                            evidence_features=evidence,
                            test=test,
                            target=sample_target(context, test, rng),
                        )
                    )
    return tuple(records)


def sample_target(context: PixelControlContext, test: str, rng: np.random.Generator) -> tuple[float, float]:
    if test in CONTROLLERS:
        success, unsafe = rollout_controller(context.target, context.regime, test, rng)
        return float(success), float(unsafe)
    return (
        float(rng.binomial(1, PROBE_LIKELIHOOD[test][context.regime])),
        float(rng.binomial(1, PROBE_UNSAFE[test])),
    )


def train_model(records: tuple[PixelControlRecord, ...], hidden_dim: int, seed: int) -> TinyMLP:
    x = np.asarray([make_input(record.image_features, record.evidence_features, record.test) for record in records])
    y = np.asarray([record.target for record in records], dtype=float)
    model = TinyMLP.create(x.shape[1], hidden_dim, y.shape[1], seed)
    model.fit(x, y, epochs=650, learning_rate=0.018)
    return model


def make_input(image: tuple[float, ...], evidence: tuple[float, ...], test: str) -> np.ndarray:
    test_one_hot = np.asarray([1.0 if test == item else 0.0 for item in TESTS], dtype=float)
    return np.concatenate([np.asarray(image, dtype=float), np.asarray(evidence, dtype=float), test_one_hot])


def unknown_evidence() -> tuple[float, ...]:
    return tuple(UNKNOWN_EVIDENCE for _ in PROBES)


def update_evidence(evidence: tuple[float, ...], probe: str, positive: bool) -> tuple[float, ...]:
    values = list(evidence)
    values[PROBES.index(probe)] = 1.0 if positive else 0.0
    return tuple(values)


def predict(model: TinyMLP, image: tuple[float, ...], evidence: tuple[float, ...], test: str) -> np.ndarray:
    return model.predict(make_input(image, evidence, test)[None, :])[0]


def evaluate_policy(
    name: str,
    contexts: tuple[PixelControlContext, ...],
    model: TinyMLP,
    max_probes: int,
    unsafe_weight: float,
    probe_weight: float,
    entropy_only: bool = False,
) -> PixelControlResult:
    totals = {"success": 0.0, "unsafe": 0.0, "probes": 0.0}
    by_regime: dict[str, dict[str, float | str]] = {}
    for regime in REGIMES:
        selected = [context for context in contexts if context.regime == regime]
        regime_totals = {"success": 0.0, "unsafe": 0.0, "probes": 0.0}
        action_weights: dict[str, float] = {}
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
            )
            for key in regime_totals:
                regime_totals[key] += float(metrics[key])
            action = str(metrics["dominant_action"])
            action_weights[action] = action_weights.get(action, 0.0) + float(metrics["dominant_weight"])
        n_regime = float(len(selected))
        for key in totals:
            totals[key] += regime_totals[key]
        by_regime[regime] = {
            "dominant_action": max(action_weights, key=action_weights.get),
            "success_rate": regime_totals["success"] / n_regime,
            "unsafe_failure_rate": regime_totals["unsafe"] / n_regime,
            "mean_probes": regime_totals["probes"] / n_regime,
        }
    n_total = float(len(contexts))
    success = totals["success"] / n_total
    unsafe = totals["unsafe"] / n_total
    probes = totals["probes"] / n_total
    return PixelControlResult(
        name=name,
        success_rate=success,
        unsafe_failure_rate=unsafe,
        mean_probes=probes,
        risk_adjusted_score=success - unsafe_weight * unsafe - probe_weight * probes,
        by_regime=by_regime,
    )


def eval_tree(
    context: PixelControlContext,
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
) -> dict[str, float | str]:
    decision = choose_decision(
        model=model,
        image=context.image_features,
        evidence=evidence,
        used=used,
        n_probes=n_probes,
        max_probes=max_probes,
        unsafe_weight=unsafe_weight,
        probe_weight=probe_weight,
        entropy_only=entropy_only,
    )
    if decision["kind"] == "act":
        controller = str(decision["choice"])
        success, unsafe = expected_controller_outcome(context, controller)
        return {
            "success": path_probability * safe_probe_probability * success,
            "unsafe": path_probability * (1.0 - safe_probe_probability * (1.0 - unsafe)),
            "probes": path_probability * n_probes,
            "dominant_action": controller,
            "dominant_weight": path_probability,
        }

    probe = str(decision["choice"])
    parts = []
    for positive in (False, True):
        p_e = PROBE_LIKELIHOOD[probe][context.regime] if positive else 1.0 - PROBE_LIKELIHOOD[probe][context.regime]
        parts.append(
            eval_tree(
                context=context,
                model=model,
                evidence=update_evidence(evidence, probe, positive),
                used=used + (probe,),
                path_probability=path_probability * p_e,
                safe_probe_probability=safe_probe_probability * (1.0 - PROBE_UNSAFE[probe]),
                n_probes=n_probes + 1,
                max_probes=max_probes,
                unsafe_weight=unsafe_weight,
                probe_weight=probe_weight,
                entropy_only=entropy_only,
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
    image: tuple[float, ...],
    evidence: tuple[float, ...],
    used: tuple[str, ...],
    n_probes: int,
    max_probes: int,
    unsafe_weight: float,
    probe_weight: float,
    entropy_only: bool,
) -> dict[str, float | str]:
    direct = direct_decision(model, image, evidence, unsafe_weight)
    if n_probes >= max_probes:
        return direct

    candidates = [direct]
    for probe in PROBES:
        if probe in used:
            continue
        pred = predict(model, image, evidence, probe)
        p_pos = float(np.clip(pred[0], 0.0, 1.0))
        p_unsafe = float(np.clip(pred[1], 0.0, 1.0))
        pos_score = model_value(
            model,
            image,
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
            image,
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
    image: tuple[float, ...],
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
        image,
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
    pred = predict(model, image, evidence, probe)
    p_pos = float(np.clip(pred[0], 0.0, 1.0))
    p_unsafe = float(np.clip(pred[1], 0.0, 1.0))
    pos_score = model_value(
        model,
        image,
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
        image,
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
    image: tuple[float, ...],
    evidence: tuple[float, ...],
    unsafe_weight: float,
) -> dict[str, float | str]:
    candidates = []
    for controller in CONTROLLERS:
        pred = predict(model, image, evidence, controller)
        score = float(pred[0] - unsafe_weight * pred[1])
        candidates.append((score, controller))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    score, controller = candidates[0]
    return {"kind": "act", "choice": controller, "score": score}


def binary_entropy(p: float) -> float:
    p = float(np.clip(p, 1e-8, 1.0 - 1e-8))
    return float(-(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p)))


@lru_cache(maxsize=None)
def cached_expected_outcome(context_id: int, target_x: float, target_y: float, regime: str, controller: str) -> tuple[float, float]:
    rng = np.random.default_rng(10_000 + context_id * 97 + CONTROLLERS.index(controller) * 17)
    successes = []
    unsafes = []
    for _ in range(64):
        success, unsafe = rollout_controller((target_x, target_y), regime, controller, rng)
        successes.append(success)
        unsafes.append(unsafe)
    return float(np.mean(successes)), float(np.mean(unsafes))


def expected_controller_outcome(context: PixelControlContext, controller: str) -> tuple[float, float]:
    return cached_expected_outcome(
        context.context_id,
        round(context.target[0], 5),
        round(context.target[1], 5),
        context.regime,
        controller,
    )


def rollout_controller(
    target: tuple[float, float],
    regime: str,
    controller: str,
    rng: np.random.Generator,
) -> tuple[bool, bool]:
    dynamics = DYNAMICS[regime]
    position = np.zeros(2, dtype=float)
    target_vec = np.asarray(target, dtype=float)
    unsafe = False
    for _ in range(controller_steps(controller)):
        remaining = target_vec - position
        if float(np.linalg.norm(remaining)) < 0.12:
            break
        action = controller_action(controller, remaining)
        unsafe = unsafe or bool(np.linalg.norm(action) > dynamics.unsafe_action_norm)
        transformed = dynamics.scale * action + rng.normal(0.0, dynamics.noise, size=2)
        position = position + transformed
    success = bool(np.linalg.norm(target_vec - position) < 0.14 and not unsafe)
    return success, unsafe


def controller_steps(controller: str) -> int:
    return {
        "fast_direct": 2,
        "slow_direct": 5,
        "stabilized": 5,
        "high_force": 3,
    }[controller]


def controller_action(controller: str, remaining: np.ndarray) -> np.ndarray:
    gain, cap = {
        "fast_direct": (1.00, 1.40),
        "slow_direct": (0.45, 0.55),
        "stabilized": (0.62, 0.72),
        "high_force": (1.75, 2.00),
    }[controller]
    action = gain * remaining
    norm = float(np.linalg.norm(action))
    if norm > cap:
        action = action * (cap / norm)
    return action


def evaluate_oracle(
    contexts: tuple[PixelControlContext, ...],
    unsafe_weight: float,
    probe_weight: float,
) -> PixelControlResult:
    by_regime: dict[str, dict[str, float | str]] = {}
    total_success = 0.0
    total_unsafe = 0.0
    for regime in REGIMES:
        selected = [context for context in contexts if context.regime == regime]
        regime_success = 0.0
        regime_unsafe = 0.0
        action_counts: dict[str, float] = {}
        for context in selected:
            utilities = {}
            outcomes = {}
            for controller in CONTROLLERS:
                success, unsafe = expected_controller_outcome(context, controller)
                utilities[controller] = success - unsafe_weight * unsafe
                outcomes[controller] = (success, unsafe)
            controller = max(utilities, key=utilities.get)
            success, unsafe = outcomes[controller]
            regime_success += success
            regime_unsafe += unsafe
            action_counts[controller] = action_counts.get(controller, 0.0) + 1.0
        total_success += regime_success
        total_unsafe += regime_unsafe
        n_regime = float(len(selected))
        by_regime[regime] = {
            "dominant_action": max(action_counts, key=action_counts.get),
            "success_rate": regime_success / n_regime,
            "unsafe_failure_rate": regime_unsafe / n_regime,
            "mean_probes": 0.0,
        }
    n_total = float(len(contexts))
    success = total_success / n_total
    unsafe = total_unsafe / n_total
    return PixelControlResult(
        name="oracle_regime",
        success_rate=success,
        unsafe_failure_rate=unsafe,
        mean_probes=0.0,
        risk_adjusted_score=success - unsafe_weight * unsafe - probe_weight * 0.0,
        by_regime=by_regime,
    )
