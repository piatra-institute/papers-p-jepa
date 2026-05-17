"""Hidden-regime manipulation world for the P-JEPA paper.

The world is intentionally small. All objects share the same visible class
(``plate``), while the hidden regime determines which task action is safe.
Safe probes produce stochastic evidence about the hidden regime.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


REGIMES = ("dry", "soapy", "cracked", "heavy")
DIRECT_ACTIONS = ("lift_fast", "lift_slow", "grip_hard", "two_contact_lift")
PROBES = ("shear_probe", "tap_probe", "weigh_probe")

PRIOR = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)


@dataclass(frozen=True)
class ActionOutcome:
    success: float
    unsafe: float


# Success is task completion probability given the hidden regime and direct
# action. Unsafe is the probability of drop, breakage, or comparable failure.
ACTION_MODEL: dict[str, dict[str, ActionOutcome]] = {
    "dry": {
        "lift_fast": ActionOutcome(success=0.96, unsafe=0.02),
        "lift_slow": ActionOutcome(success=0.88, unsafe=0.01),
        "grip_hard": ActionOutcome(success=0.94, unsafe=0.02),
        "two_contact_lift": ActionOutcome(success=0.84, unsafe=0.01),
    },
    "soapy": {
        "lift_fast": ActionOutcome(success=0.18, unsafe=0.75),
        "lift_slow": ActionOutcome(success=0.62, unsafe=0.28),
        "grip_hard": ActionOutcome(success=0.42, unsafe=0.48),
        "two_contact_lift": ActionOutcome(success=0.90, unsafe=0.06),
    },
    "cracked": {
        "lift_fast": ActionOutcome(success=0.36, unsafe=0.52),
        "lift_slow": ActionOutcome(success=0.86, unsafe=0.08),
        "grip_hard": ActionOutcome(success=0.12, unsafe=0.84),
        "two_contact_lift": ActionOutcome(success=0.55, unsafe=0.35),
    },
    "heavy": {
        "lift_fast": ActionOutcome(success=0.30, unsafe=0.60),
        "lift_slow": ActionOutcome(success=0.48, unsafe=0.42),
        "grip_hard": ActionOutcome(success=0.91, unsafe=0.05),
        "two_contact_lift": ActionOutcome(success=0.72, unsafe=0.18),
    },
}


# P(positive evidence | hidden regime, probe). Probes are safe but imperfect.
PROBE_LIKELIHOOD: dict[str, dict[str, float]] = {
    "shear_probe": {"dry": 0.04, "soapy": 0.94, "cracked": 0.06, "heavy": 0.04},
    "tap_probe": {"dry": 0.04, "soapy": 0.06, "cracked": 0.93, "heavy": 0.08},
    "weigh_probe": {"dry": 0.05, "soapy": 0.05, "cracked": 0.07, "heavy": 0.95},
}


PROBE_UNSAFE: dict[str, float] = {
    "shear_probe": 0.005,
    "tap_probe": 0.010,
    "weigh_probe": 0.005,
}


def prediction_matrix() -> np.ndarray:
    """Rows are hidden regimes, columns are direct-action success predictions."""
    return np.array(
        [
            [ACTION_MODEL[r][a].success for a in DIRECT_ACTIONS]
            for r in REGIMES
        ],
        dtype=float,
    )


def unsafe_matrix() -> np.ndarray:
    """Rows are hidden regimes, columns are direct-action unsafe probabilities."""
    return np.array(
        [
            [ACTION_MODEL[r][a].unsafe for a in DIRECT_ACTIONS]
            for r in REGIMES
        ],
        dtype=float,
    )


def bayes_update(posterior: np.ndarray, probe: str, positive: bool) -> np.ndarray:
    likelihood = np.array([PROBE_LIKELIHOOD[probe][r] for r in REGIMES], dtype=float)
    if not positive:
        likelihood = 1.0 - likelihood
    updated = posterior * likelihood
    total = float(updated.sum())
    if total <= 0:
        return posterior.copy()
    return updated / total


def evidence_probability(posterior: np.ndarray, probe: str, positive: bool) -> float:
    likelihood = np.array([PROBE_LIKELIHOOD[probe][r] for r in REGIMES], dtype=float)
    if not positive:
        likelihood = 1.0 - likelihood
    return float(np.dot(posterior, likelihood))


def obstruction(posterior: np.ndarray) -> float:
    """Weighted section disagreement over direct-action predictions.

    This is the degree-zero sheaf energy used in the prototype:
    the posterior-weighted variance of local predictive sections.
    """
    preds = prediction_matrix()
    mean = posterior @ preds
    diffs = preds - mean
    return float(np.sum(posterior[:, None] * diffs * diffs))


def best_direct_action(posterior: np.ndarray) -> str:
    expected_success = posterior @ prediction_matrix()
    return DIRECT_ACTIONS[int(np.argmax(expected_success))]


def expected_action_outcome(posterior: np.ndarray, action: str) -> ActionOutcome:
    success = 0.0
    unsafe = 0.0
    for i, regime in enumerate(REGIMES):
        outcome = ACTION_MODEL[regime][action]
        success += posterior[i] * outcome.success
        unsafe += posterior[i] * outcome.unsafe
    return ActionOutcome(success=float(success), unsafe=float(unsafe))
