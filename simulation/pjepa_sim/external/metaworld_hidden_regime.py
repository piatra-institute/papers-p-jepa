"""Meta-World hidden-regime adapter.

The synthetic DishWorld benchmark proves the P-JEPA decision rule by exact
enumeration. This module defines the first external benchmark bridge: a
Meta-World task is wrapped in hidden action regimes, safe probes, posterior
updates, and obstruction measurement.

The module is importable without Meta-World installed. The heavy imports happen
only inside ``make_metaworld_env`` and ``check_metaworld``.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


DEFAULT_ENV_ID = "Meta-World/MT1"
DEFAULT_TASK = "reach-v3"
REGIMES = ("nominal", "slippery", "fragile", "heavy")
PROBES = ("shear_probe", "tap_probe", "weigh_probe")
PROBE_STRATEGIES = ("none", "random", "entropy", "obstruction", "oracle")
POLICIES = ("random", "reach_scripted", "reach_belief_safe")
DEFAULT_STRATEGIES = (
    "fixed_no_probe",
    "belief_safe_no_probe",
    "random_one_probe_belief_safe",
    "random_probe_belief_safe",
    "entropy_probe_belief_safe",
    "obstruction_probe_belief_safe",
    "oracle_belief_safe",
)


@dataclass(frozen=True)
class ExternalAvailability:
    name: str
    available: bool
    error: str | None
    install_hint: str
    api_hint: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "error": self.error,
            "install_hint": self.install_hint,
            "api_hint": self.api_hint,
        }


@dataclass(frozen=True)
class HiddenRegime:
    name: str
    action_scale: float = 1.0
    action_noise: float = 0.0
    unsafe_action_norm: float = 2.0
    unsafe_penalty: float = 0.0

    def as_dict(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "action_scale": self.action_scale,
            "action_noise": self.action_noise,
            "unsafe_action_norm": self.unsafe_action_norm,
            "unsafe_penalty": self.unsafe_penalty,
        }


@dataclass(frozen=True)
class ProbeSpec:
    name: str
    likelihood: dict[str, float]
    unsafe_probability: float
    reward_cost: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "likelihood": dict(self.likelihood),
            "unsafe_probability": self.unsafe_probability,
            "reward_cost": self.reward_cost,
        }


@dataclass(frozen=True)
class MetaWorldHiddenRegimeConfig:
    env_id: str = DEFAULT_ENV_ID
    task: str = DEFAULT_TASK
    seed: int = 0
    max_episode_steps: int = 150
    obstruction_threshold: float = 0.150
    max_probes: int = 3
    success_info_key: str = "success"
    prior: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
    regimes: dict[str, HiddenRegime] = field(default_factory=dict)
    probes: dict[str, ProbeSpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.regimes:
            object.__setattr__(self, "regimes", default_regimes())
        if not self.probes:
            object.__setattr__(self, "probes", default_probes())
        if set(self.regimes) != set(REGIMES):
            raise ValueError(f"regimes must be exactly {REGIMES}")
        if set(self.probes) != set(PROBES):
            raise ValueError(f"probes must be exactly {PROBES}")
        prior = np.asarray(self.prior, dtype=float)
        if prior.shape != (len(REGIMES),) or float(prior.sum()) <= 0:
            raise ValueError("prior must contain one positive weight per regime")

    def prior_array(self) -> np.ndarray:
        prior = np.asarray(self.prior, dtype=float)
        return prior / prior.sum()

    def as_dict(self) -> dict[str, Any]:
        return {
            "env_id": self.env_id,
            "task": self.task,
            "seed": self.seed,
            "max_episode_steps": self.max_episode_steps,
            "obstruction_threshold": self.obstruction_threshold,
            "max_probes": self.max_probes,
            "success_info_key": self.success_info_key,
            "prior": {name: float(value) for name, value in zip(REGIMES, self.prior_array())},
            "regimes": {name: regime.as_dict() for name, regime in self.regimes.items()},
            "probes": {name: probe.as_dict() for name, probe in self.probes.items()},
        }


class Policy(Protocol):
    def reset(self, observation: np.ndarray, info: dict[str, Any]) -> None:
        ...

    def act(self, observation: np.ndarray, posterior: np.ndarray) -> np.ndarray:
        ...


@dataclass
class EpisodeResult:
    regime: str
    success: bool
    unsafe: bool
    total_reward: float
    steps: int
    probes: list[dict[str, Any]]
    obstruction_start: float
    obstruction_at_action: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime,
            "success": self.success,
            "unsafe": self.unsafe,
            "total_reward": self.total_reward,
            "steps": self.steps,
            "probes": list(self.probes),
            "obstruction_start": self.obstruction_start,
            "obstruction_at_action": self.obstruction_at_action,
        }


class RandomPolicy:
    """Smoke-test policy for adapter plumbing, not a meaningful baseline."""

    def __init__(self, action_space: Any, rng: np.random.Generator):
        self.action_space = action_space
        self.rng = rng

    def reset(self, observation: np.ndarray, info: dict[str, Any]) -> None:
        return None

    def act(self, observation: np.ndarray, posterior: np.ndarray) -> np.ndarray:
        if hasattr(self.action_space, "sample"):
            return np.asarray(self.action_space.sample(), dtype=float)
        raise TypeError("action_space must provide sample()")


class ScriptedReachPolicy:
    """Simple controller for Meta-World reach-v3 smoke tests.

    The policy uses the public observation for the hand position and the
    environment target position. It is not a learned baseline and should not be
    treated as a P-JEPA result.
    """

    def __init__(self, env: Any, gain: float = 8.0):
        self.env = env
        self.gain = gain
        self.target = np.zeros(3, dtype=float)

    def reset(self, observation: np.ndarray, info: dict[str, Any]) -> None:
        unwrapped = getattr(self.env, "unwrapped", self.env)
        if hasattr(unwrapped, "_target_pos"):
            self.target = np.asarray(getattr(unwrapped, "_target_pos"), dtype=float)
        elif hasattr(unwrapped, "goal"):
            self.target = np.asarray(getattr(unwrapped, "goal"), dtype=float)
        else:
            self.target = np.asarray(observation[:3], dtype=float)

    def act(self, observation: np.ndarray, posterior: np.ndarray) -> np.ndarray:
        action_shape = self.env.action_space.shape
        action = np.zeros(action_shape, dtype=float)
        hand = np.asarray(observation[:3], dtype=float)
        action[:3] = np.clip(self.gain * (self.target - hand), -1.0, 1.0)
        return action


class BeliefSafeReachPolicy(ScriptedReachPolicy):
    """Reach controller that lowers action magnitude when posterior risk is high."""

    def __init__(
        self,
        env: Any,
        gain: float = 8.0,
        fragile_threshold: float = 0.45,
        slippery_threshold: float = 0.60,
        fragile_cap: float = 0.86,
        slippery_cap: float = 1.10,
    ):
        super().__init__(env, gain=gain)
        self.fragile_threshold = fragile_threshold
        self.slippery_threshold = slippery_threshold
        self.fragile_cap = fragile_cap
        self.slippery_cap = slippery_cap

    def act(self, observation: np.ndarray, posterior: np.ndarray) -> np.ndarray:
        action = super().act(observation, posterior)
        cap = self._norm_cap(posterior)
        norm = float(np.linalg.norm(action))
        if norm > cap:
            action = action * (cap / norm)
        return action

    def _norm_cap(self, posterior: np.ndarray) -> float:
        posterior_by_regime = posterior_dict(posterior)
        if posterior_by_regime["fragile"] >= self.fragile_threshold:
            return self.fragile_cap
        if posterior_by_regime["slippery"] >= self.slippery_threshold:
            return self.slippery_cap
        return 10.0


@dataclass(frozen=True)
class StrategySpec:
    name: str
    policy_name: str
    probe_strategy: str
    max_probes: int | None = None

    def as_dict(self) -> dict[str, str | int | None]:
        return {
            "name": self.name,
            "policy_name": self.policy_name,
            "probe_strategy": self.probe_strategy,
            "max_probes": self.max_probes,
        }


class HiddenRegimeMetaWorldAdapter:
    """Apply hidden regimes, probes, and obstruction to a Meta-World env."""

    def __init__(
        self,
        env: Any,
        config: MetaWorldHiddenRegimeConfig | None = None,
        model_config: MetaWorldHiddenRegimeConfig | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.env = env
        self.config = config or MetaWorldHiddenRegimeConfig()
        self.model_config = model_config or self.config
        self.rng = rng or np.random.default_rng(self.config.seed)
        self.posterior = self.model_config.prior_array()
        self.hidden_regime = REGIMES[0]
        self.unsafe = False

    def reset(self, regime: str | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        self.posterior = self.model_config.prior_array()
        self.hidden_regime = regime or str(self.rng.choice(REGIMES, p=self.posterior))
        self.unsafe = False
        observation, info = self.env.reset()
        info = dict(info)
        info["hidden_regime"] = self.hidden_regime
        info["posterior"] = posterior_dict(self.posterior)
        info["obstruction"] = obstruction(self.model_config, self.posterior)
        return np.asarray(observation, dtype=float), info

    def run_probe(self, probe_name: str) -> dict[str, Any]:
        probe = self.config.probes[probe_name]
        likelihood = probe.likelihood[self.hidden_regime]
        positive = bool(self.rng.random() < likelihood)
        probe_unsafe = bool(self.rng.random() < probe.unsafe_probability)
        self.unsafe = self.unsafe or probe_unsafe
        self.posterior = bayes_update(self.model_config, self.posterior, probe_name, positive)
        return {
            "probe": probe_name,
            "evidence": "positive" if positive else "negative",
            "unsafe": probe_unsafe,
            "posterior": posterior_dict(self.posterior),
            "obstruction": obstruction(self.model_config, self.posterior),
            "reward_cost": probe.reward_cost,
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        transformed = self.transform_action(action)
        unsafe_step = self.is_unsafe_action(transformed)
        self.unsafe = self.unsafe or unsafe_step
        observation, reward, terminated, truncated, info = self.env.step(transformed)
        regime = self.config.regimes[self.hidden_regime]
        adjusted_reward = float(reward) - (regime.unsafe_penalty if unsafe_step else 0.0)
        info = dict(info)
        info["hidden_regime"] = self.hidden_regime
        info["unsafe"] = unsafe_step
        info["posterior"] = posterior_dict(self.posterior)
        info["obstruction"] = obstruction(self.model_config, self.posterior)
        return np.asarray(observation, dtype=float), adjusted_reward, bool(terminated), bool(truncated), info

    def transform_action(self, action: np.ndarray) -> np.ndarray:
        regime = self.config.regimes[self.hidden_regime]
        action = np.asarray(action, dtype=float)
        noise = self.rng.normal(0.0, regime.action_noise, size=action.shape)
        transformed = regime.action_scale * action + noise
        if hasattr(self.env, "action_space"):
            low = getattr(self.env.action_space, "low", None)
            high = getattr(self.env.action_space, "high", None)
            if low is not None and high is not None:
                transformed = np.clip(transformed, low, high)
        return transformed

    def is_unsafe_action(self, action: np.ndarray) -> bool:
        threshold = self.config.regimes[self.hidden_regime].unsafe_action_norm
        return bool(np.linalg.norm(action) > threshold)


def default_regimes() -> dict[str, HiddenRegime]:
    return {
        "nominal": HiddenRegime(
            name="nominal",
            action_scale=1.0,
            action_noise=0.0,
            unsafe_action_norm=2.0,
            unsafe_penalty=0.0,
        ),
        "slippery": HiddenRegime(
            name="slippery",
            action_scale=0.75,
            action_noise=0.12,
            unsafe_action_norm=1.30,
            unsafe_penalty=0.30,
        ),
        "fragile": HiddenRegime(
            name="fragile",
            action_scale=1.0,
            action_noise=0.02,
            unsafe_action_norm=0.95,
            unsafe_penalty=0.70,
        ),
        "heavy": HiddenRegime(
            name="heavy",
            action_scale=0.52,
            action_noise=0.03,
            unsafe_action_norm=1.85,
            unsafe_penalty=0.15,
        ),
    }


def default_probes() -> dict[str, ProbeSpec]:
    return {
        "shear_probe": ProbeSpec(
            name="shear_probe",
            likelihood={"nominal": 0.06, "slippery": 0.92, "fragile": 0.10, "heavy": 0.10},
            unsafe_probability=0.01,
            reward_cost=0.02,
        ),
        "tap_probe": ProbeSpec(
            name="tap_probe",
            likelihood={"nominal": 0.06, "slippery": 0.10, "fragile": 0.90, "heavy": 0.12},
            unsafe_probability=0.02,
            reward_cost=0.03,
        ),
        "weigh_probe": ProbeSpec(
            name="weigh_probe",
            likelihood={"nominal": 0.08, "slippery": 0.08, "fragile": 0.12, "heavy": 0.90},
            unsafe_probability=0.01,
            reward_cost=0.02,
        ),
    }


def check_metaworld() -> ExternalAvailability:
    try:
        importlib.import_module("gymnasium")
        importlib.import_module("metaworld")
    except Exception as exc:
        return ExternalAvailability(
            name="metaworld",
            available=False,
            error=f"{type(exc).__name__}: {exc}",
            install_hint="uv add gymnasium metaworld",
            api_hint="gym.make('Meta-World/MT1', env_name='reach-v3', seed=0)",
        )
    return ExternalAvailability(
        name="metaworld",
        available=True,
        error=None,
        install_hint="already importable",
        api_hint="gym.make('Meta-World/MT1', env_name='reach-v3', seed=0)",
    )


def make_metaworld_env(config: MetaWorldHiddenRegimeConfig) -> Any:
    gym = importlib.import_module("gymnasium")
    importlib.import_module("metaworld")
    return gym.make(
        config.env_id,
        env_name=config.task,
        seed=config.seed,
        max_episode_steps=config.max_episode_steps,
    )


def posterior_dict(posterior: np.ndarray) -> dict[str, float]:
    return {regime: float(value) for regime, value in zip(REGIMES, posterior)}


def prediction_matrix(config: MetaWorldHiddenRegimeConfig) -> np.ndarray:
    return np.array(
        [
            [
                config.regimes[regime].action_scale,
                config.regimes[regime].action_noise,
                config.regimes[regime].unsafe_action_norm,
            ]
            for regime in REGIMES
        ],
        dtype=float,
    )


def obstruction(config: MetaWorldHiddenRegimeConfig, posterior: np.ndarray) -> float:
    preds = prediction_matrix(config)
    mean = posterior @ preds
    diffs = preds - mean
    return float(np.sum(posterior[:, None] * diffs * diffs))


def bayes_update(
    config: MetaWorldHiddenRegimeConfig,
    posterior: np.ndarray,
    probe_name: str,
    positive: bool,
) -> np.ndarray:
    probe = config.probes[probe_name]
    likelihood = np.array([probe.likelihood[regime] for regime in REGIMES], dtype=float)
    if not positive:
        likelihood = 1.0 - likelihood
    updated = posterior * likelihood
    total = float(updated.sum())
    if total <= 0:
        return posterior.copy()
    return updated / total


def evidence_probability(
    config: MetaWorldHiddenRegimeConfig,
    posterior: np.ndarray,
    probe_name: str,
    positive: bool,
) -> float:
    probe = config.probes[probe_name]
    likelihood = np.array([probe.likelihood[regime] for regime in REGIMES], dtype=float)
    if not positive:
        likelihood = 1.0 - likelihood
    return float(posterior @ likelihood)


def expected_probe_reduction(
    config: MetaWorldHiddenRegimeConfig,
    posterior: np.ndarray,
    probe_name: str,
) -> float:
    before = obstruction(config, posterior)
    after = 0.0
    for positive in (False, True):
        p_e = evidence_probability(config, posterior, probe_name, positive)
        after += p_e * obstruction(config, bayes_update(config, posterior, probe_name, positive))
    return before - after


def posterior_entropy(posterior: np.ndarray) -> float:
    positive = posterior[posterior > 0.0]
    return float(-np.sum(positive * np.log2(positive)))


def expected_entropy_reduction(
    config: MetaWorldHiddenRegimeConfig,
    posterior: np.ndarray,
    probe_name: str,
) -> float:
    before = posterior_entropy(posterior)
    after = 0.0
    for positive in (False, True):
        p_e = evidence_probability(config, posterior, probe_name, positive)
        after += p_e * posterior_entropy(bayes_update(config, posterior, probe_name, positive))
    return before - after


def choose_probe(
    config: MetaWorldHiddenRegimeConfig,
    posterior: np.ndarray,
    used: set[str],
) -> str | None:
    candidates = [probe for probe in PROBES if probe not in used]
    if not candidates:
        return None
    scores = [(expected_probe_reduction(config, posterior, probe), probe) for probe in candidates]
    scores.sort(key=lambda item: (-item[0], item[1]))
    return scores[0][1]


def choose_entropy_probe(
    config: MetaWorldHiddenRegimeConfig,
    posterior: np.ndarray,
    used: set[str],
) -> str | None:
    candidates = [probe for probe in PROBES if probe not in used]
    if not candidates:
        return None
    scores = [(expected_entropy_reduction(config, posterior, probe), probe) for probe in candidates]
    scores.sort(key=lambda item: (-item[0], item[1]))
    return scores[0][1]


def choose_random_probe(
    config: MetaWorldHiddenRegimeConfig,
    used: set[str],
    rng: np.random.Generator,
) -> str | None:
    candidates = [probe for probe in PROBES if probe not in used]
    if not candidates:
        return None
    return str(rng.choice(candidates))


def choose_preflight_probes(config: MetaWorldHiddenRegimeConfig) -> list[dict[str, Any]]:
    posterior = config.prior_array()
    used: set[str] = set()
    trace: list[dict[str, Any]] = []
    while obstruction(config, posterior) > config.obstruction_threshold and len(used) < config.max_probes:
        probe = choose_probe(config, posterior, used)
        if probe is None:
            break
        before = obstruction(config, posterior)
        expected_reduction = expected_probe_reduction(config, posterior, probe)
        used.add(probe)
        trace.append(
            {
                "probe": probe,
                "obstruction_before": before,
                "expected_obstruction_reduction": expected_reduction,
                "posterior_before": posterior_dict(posterior),
            }
        )
        # The protocol trace is evidence-free, so use the most likely evidence
        # under the current prior to show the deterministic probe order.
        positive = evidence_probability(config, posterior, probe, True) >= 0.5
        posterior = bayes_update(config, posterior, probe, positive)
    return trace


def protocol_summary(config: MetaWorldHiddenRegimeConfig | None = None) -> dict[str, Any]:
    config = config or MetaWorldHiddenRegimeConfig()
    prior = config.prior_array()
    return {
        "benchmark": "metaworld_hidden_regime",
        "status": "protocol_defined",
        "config": config.as_dict(),
        "obstruction_prior": obstruction(config, prior),
        "preflight_probe_order": choose_preflight_probes(config),
        "notes": [
            "This is an external benchmark adapter, not a completed external result.",
            "Real scores require installing Meta-World/MuJoCo and supplying a task policy.",
            "The wrapper keeps the hidden regime out of the observation and exposes it only in logs.",
        ],
    }


def run_smoke(
    config: MetaWorldHiddenRegimeConfig,
    episodes: int,
    policy_name: str = "reach_scripted",
    probe_strategy: str = "obstruction",
    model_config: MetaWorldHiddenRegimeConfig | None = None,
) -> dict[str, Any]:
    env = make_metaworld_env(config)
    rng = np.random.default_rng(config.seed)
    adapter = HiddenRegimeMetaWorldAdapter(env, config=config, model_config=model_config, rng=rng)
    policy = make_policy(policy_name, env, rng)
    results = []
    try:
        for _ in range(episodes):
            results.append(
                run_episode(
                    adapter,
                    policy,
                    probe_strategy=probe_strategy,
                ).as_dict()
            )
    finally:
        env.close()
    return {
        "benchmark": "metaworld_hidden_regime_smoke",
        "policy": policy_name,
        "probe_strategy": probe_strategy,
        "episodes": episodes,
        "aggregate": aggregate_episode_dicts(results),
        "results": results,
    }


def run_random_smoke(
    config: MetaWorldHiddenRegimeConfig,
    episodes: int,
) -> dict[str, Any]:
    return run_smoke(config, episodes=episodes, policy_name="random")


def make_policy(policy_name: str, env: Any, rng: np.random.Generator) -> Policy:
    if policy_name == "random":
        return RandomPolicy(env.action_space, rng)
    if policy_name == "reach_scripted":
        return ScriptedReachPolicy(env)
    if policy_name == "reach_belief_safe":
        return BeliefSafeReachPolicy(env)
    raise ValueError(f"unknown external smoke policy {policy_name!r}")


def strategy_specs(names: list[str] | None = None) -> list[StrategySpec]:
    specs = {
        "fixed_no_probe": StrategySpec(
            name="fixed_no_probe",
            policy_name="reach_scripted",
            probe_strategy="none",
        ),
        "belief_safe_no_probe": StrategySpec(
            name="belief_safe_no_probe",
            policy_name="reach_belief_safe",
            probe_strategy="none",
        ),
        "random_one_probe_belief_safe": StrategySpec(
            name="random_one_probe_belief_safe",
            policy_name="reach_belief_safe",
            probe_strategy="random",
            max_probes=1,
        ),
        "random_probe_belief_safe": StrategySpec(
            name="random_probe_belief_safe",
            policy_name="reach_belief_safe",
            probe_strategy="random",
        ),
        "entropy_probe_belief_safe": StrategySpec(
            name="entropy_probe_belief_safe",
            policy_name="reach_belief_safe",
            probe_strategy="entropy",
        ),
        "obstruction_probe_belief_safe": StrategySpec(
            name="obstruction_probe_belief_safe",
            policy_name="reach_belief_safe",
            probe_strategy="obstruction",
        ),
        "oracle_belief_safe": StrategySpec(
            name="oracle_belief_safe",
            policy_name="reach_belief_safe",
            probe_strategy="oracle",
        ),
    }
    selected = list(DEFAULT_STRATEGIES) if names is None else names
    unknown = [name for name in selected if name not in specs]
    if unknown:
        raise ValueError(f"unknown strategy: {', '.join(unknown)}")
    return [specs[name] for name in selected]


def run_strategy_benchmark(
    config: MetaWorldHiddenRegimeConfig,
    episodes: int,
    strategy_names: list[str] | None = None,
    model_config: MetaWorldHiddenRegimeConfig | None = None,
) -> dict[str, Any]:
    regime_schedule = balanced_regime_schedule(config, episodes)
    strategy_results: dict[str, Any] = {}
    for spec in strategy_specs(strategy_names):
        env = make_metaworld_env(config)
        rng = np.random.default_rng(config.seed)
        adapter = HiddenRegimeMetaWorldAdapter(env, config=config, model_config=model_config, rng=rng)
        policy = make_policy(spec.policy_name, env, rng)
        episode_results = []
        try:
            for regime in regime_schedule:
                episode_results.append(
                    run_episode(
                        adapter,
                        policy,
                        probe_strategy=spec.probe_strategy,
                        regime=regime,
                        max_probes=spec.max_probes,
                    ).as_dict()
                )
        finally:
            env.close()
        strategy_results[spec.name] = {
            "strategy": spec.as_dict(),
            "aggregate": aggregate_episode_dicts(episode_results),
            "results": episode_results,
        }
    return {
        "benchmark": "metaworld_hidden_regime_strategy_comparison",
        "task": config.task,
        "episodes_per_strategy": episodes,
        "model_source": "configured" if model_config is None else "learned",
        "regime_schedule": list(regime_schedule),
        "strategies": strategy_results,
    }


def balanced_regime_schedule(
    config: MetaWorldHiddenRegimeConfig,
    episodes: int,
) -> tuple[str, ...]:
    rng = np.random.default_rng(config.seed)
    base = list(REGIMES) * (episodes // len(REGIMES))
    remainder = list(rng.choice(REGIMES, size=episodes % len(REGIMES), replace=False))
    schedule = base + remainder
    rng.shuffle(schedule)
    return tuple(str(regime) for regime in schedule)


def aggregate_episode_dicts(results: list[dict[str, Any]]) -> dict[str, float]:
    if not results:
        return {
            "success_rate": 0.0,
            "unsafe_rate": 0.0,
            "risk_adjusted_score": 0.0,
            "mean_total_reward": 0.0,
            "mean_steps": 0.0,
            "mean_probes": 0.0,
            "mean_obstruction_start": 0.0,
            "mean_obstruction_at_action": 0.0,
            "mean_obstruction_reduction": 0.0,
        }
    n = float(len(results))
    obstruction_start = sum(float(result["obstruction_start"]) for result in results) / n
    obstruction_at_action = sum(float(result["obstruction_at_action"]) for result in results) / n
    return {
        "success_rate": sum(1.0 for result in results if result["success"]) / n,
        "unsafe_rate": sum(1.0 for result in results if result["unsafe"]) / n,
        "risk_adjusted_score": (
            sum(1.0 for result in results if result["success"]) / n
            - 2.0 * sum(1.0 for result in results if result["unsafe"]) / n
            - 0.02 * sum(float(len(result["probes"])) for result in results) / n
        ),
        "mean_total_reward": sum(float(result["total_reward"]) for result in results) / n,
        "mean_steps": sum(float(result["steps"]) for result in results) / n,
        "mean_probes": sum(float(len(result["probes"])) for result in results) / n,
        "mean_obstruction_start": obstruction_start,
        "mean_obstruction_at_action": obstruction_at_action,
        "mean_obstruction_reduction": obstruction_start - obstruction_at_action,
    }


def run_episode(
    adapter: HiddenRegimeMetaWorldAdapter,
    policy: Policy,
    probe_strategy: str = "obstruction",
    regime: str | None = None,
    max_probes: int | None = None,
) -> EpisodeResult:
    if probe_strategy not in PROBE_STRATEGIES:
        raise ValueError(f"unknown probe strategy {probe_strategy!r}")
    observation, info = adapter.reset(regime=regime)
    if probe_strategy == "oracle":
        adapter.posterior = one_hot_posterior(adapter.hidden_regime)
    policy.reset(observation, info)
    total_reward = 0.0
    probes: list[dict[str, Any]] = []
    used: set[str] = set()
    obstruction_start = obstruction(adapter.model_config, adapter.posterior)

    probe_limit = adapter.config.max_probes if max_probes is None else max_probes
    while should_probe(adapter.model_config, adapter.posterior, used, probe_strategy, probe_limit):
        probe = next_probe(adapter.model_config, adapter.posterior, used, probe_strategy, adapter.rng)
        if probe is None:
            break
        used.add(probe)
        event = adapter.run_probe(probe)
        total_reward -= float(event["reward_cost"])
        probes.append(event)

    obstruction_at_action = obstruction(adapter.model_config, adapter.posterior)
    success = False
    steps = 0
    terminated = False
    truncated = False
    while not (terminated or truncated) and steps < adapter.config.max_episode_steps:
        action = policy.act(observation, adapter.posterior)
        observation, reward, terminated, truncated, info = adapter.step(action)
        total_reward += reward
        steps += 1
        success = success or bool(info.get(adapter.config.success_info_key, False))

    return EpisodeResult(
        regime=adapter.hidden_regime,
        success=success,
        unsafe=adapter.unsafe,
        total_reward=float(total_reward),
        steps=steps,
        probes=probes,
        obstruction_start=obstruction_start,
        obstruction_at_action=obstruction_at_action,
    )


def one_hot_posterior(regime: str) -> np.ndarray:
    posterior = np.zeros(len(REGIMES), dtype=float)
    posterior[REGIMES.index(regime)] = 1.0
    return posterior


def should_probe(
    config: MetaWorldHiddenRegimeConfig,
    posterior: np.ndarray,
    used: set[str],
    probe_strategy: str,
    probe_limit: int,
) -> bool:
    if probe_strategy in ("none", "oracle"):
        return False
    if len(used) >= probe_limit:
        return False
    if probe_strategy == "random":
        return True
    return obstruction(config, posterior) > config.obstruction_threshold


def next_probe(
    config: MetaWorldHiddenRegimeConfig,
    posterior: np.ndarray,
    used: set[str],
    probe_strategy: str,
    rng: np.random.Generator,
) -> str | None:
    if probe_strategy == "random":
        return choose_random_probe(config, used, rng)
    if probe_strategy == "entropy":
        return choose_entropy_probe(config, posterior, used)
    if probe_strategy == "obstruction":
        return choose_probe(config, posterior, used)
    return None
