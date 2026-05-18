"""Dishworld-style data for the JEPA toy.

Produces:
- ContextSample: a 4-dim sensor vector + 2-dim visual vector + regime label.
- MaskedPair: (context_view, full_view, mask_vector). The "view" is the
  context with masked dimensions zero-filled; the full is the unmasked
  version. The JEPA loss asks the predictor to recover the full from
  the view + mask indicator.
- InterventionSample: (context, test_id, outcome_vector). For the
  intervention loss.
- OverlapPair: two contexts from the same regime with different
  noise realizations. For the bisimulation loss (paired in-regime is
  positive, cross-regime is negative).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pjepa_sim.core.dishworld import (
    ACTION_MODEL,
    DIRECT_ACTIONS,
    PROBE_LIKELIHOOD,
    PROBE_UNSAFE,
    PROBES,
    REGIMES,
)
from pjepa_sim.representation.learning import TEST_VISUAL, TRAIN_VISUAL, VISUALS


TESTS = DIRECT_ACTIONS + PROBES
N_REGIMES = len(REGIMES)
N_DIRECT_ACTIONS = len(DIRECT_ACTIONS)
N_TESTS = len(TESTS)
SENSOR_DIM = 4
VISUAL_DIM = len(VISUALS)
CONTEXT_DIM = SENSOR_DIM + VISUAL_DIM

SENSOR_BASE: dict[str, tuple[float, ...]] = {
    "dry": (0.10, 0.08, 0.18, 0.15),
    "soapy": (0.92, 0.10, 0.20, 0.85),
    "cracked": (0.12, 0.90, 0.23, 0.20),
    "heavy": (0.18, 0.22, 0.94, 0.25),
}


@dataclass(frozen=True)
class ContextSample:
    context_id: int
    regime: str
    regime_index: int
    split: str
    features: np.ndarray  # (CONTEXT_DIM,)
    sensor: np.ndarray    # (SENSOR_DIM,)
    visual: np.ndarray    # (VISUAL_DIM,)


@dataclass(frozen=True)
class MaskedPair:
    context_id: int
    full: np.ndarray  # (CONTEXT_DIM,)
    view: np.ndarray  # (CONTEXT_DIM,) masked-out dims = 0
    mask: np.ndarray  # (CONTEXT_DIM,) 1.0 where visible to encoder, 0.0 where hidden


@dataclass(frozen=True)
class InterventionSample:
    context_id: int
    regime_index: int
    features: np.ndarray  # (CONTEXT_DIM,) - full context
    test_index: int       # which test in TESTS
    outcome: np.ndarray   # length-2 vector (success, unsafe) for direct actions; length-1 for probes (padded to 2 with zero)


@dataclass(frozen=True)
class OverlapPair:
    """Two contexts from the same regime (positive overlap)."""
    a: ContextSample
    b: ContextSample


def visual_for(regime: str, split: str) -> str:
    mapping = TRAIN_VISUAL if split == "train" else TEST_VISUAL
    return mapping[regime]


def visual_to_vector(visual: str) -> np.ndarray:
    return np.asarray([1.0 if visual == v else 0.0 for v in VISUALS], dtype=float)


def generate_contexts(
    split: str,
    contexts_per_regime: int,
    sensor_noise: float,
    rng: np.random.Generator,
    base_id: int = 0,
) -> list[ContextSample]:
    contexts: list[ContextSample] = []
    cid = base_id
    for regime_index, regime in enumerate(REGIMES):
        for _ in range(contexts_per_regime):
            visual = visual_for(regime, split)
            visual_vec = visual_to_vector(visual)
            sensor = np.asarray(SENSOR_BASE[regime], dtype=float) + rng.normal(
                0.0, sensor_noise, size=SENSOR_DIM
            )
            sensor = np.clip(sensor, 0.0, 1.0)
            features = np.concatenate([sensor, visual_vec])
            contexts.append(
                ContextSample(
                    context_id=cid,
                    regime=regime,
                    regime_index=regime_index,
                    split=split,
                    features=features,
                    sensor=sensor,
                    visual=visual_vec,
                )
            )
            cid += 1
    return contexts


def random_mask(
    rng: np.random.Generator,
    mask_fraction: float,
) -> np.ndarray:
    """Random binary mask. 1.0 where visible, 0.0 where hidden.
    Mask covers approximately `mask_fraction` of CONTEXT_DIM dims."""
    n_hide = max(1, int(round(mask_fraction * CONTEXT_DIM)))
    indices = rng.permutation(CONTEXT_DIM)
    mask = np.ones(CONTEXT_DIM, dtype=float)
    mask[indices[:n_hide]] = 0.0
    return mask


def masked_pair(
    context: ContextSample,
    mask: np.ndarray,
) -> MaskedPair:
    view = context.features * mask
    return MaskedPair(
        context_id=context.context_id,
        full=context.features.copy(),
        view=view,
        mask=mask,
    )


def generate_masked_pairs(
    contexts: list[ContextSample],
    pairs_per_context: int,
    mask_fraction: float,
    rng: np.random.Generator,
) -> list[MaskedPair]:
    pairs: list[MaskedPair] = []
    for context in contexts:
        for _ in range(pairs_per_context):
            mask = random_mask(rng, mask_fraction)
            pairs.append(masked_pair(context, mask))
    return pairs


def sample_outcome(regime: str, test: str, rng: np.random.Generator) -> np.ndarray:
    if test in DIRECT_ACTIONS:
        outcome = ACTION_MODEL[regime][test]
        return np.asarray(
            [rng.binomial(1, outcome.success), rng.binomial(1, outcome.unsafe)],
            dtype=float,
        )
    likelihood = PROBE_LIKELIHOOD[test][regime]
    return np.asarray([rng.binomial(1, likelihood), 0.0], dtype=float)


def generate_interventions(
    contexts: list[ContextSample],
    repeats_per_test: int,
    rng: np.random.Generator,
) -> list[InterventionSample]:
    samples: list[InterventionSample] = []
    for context in contexts:
        for test_index, test in enumerate(TESTS):
            for _ in range(repeats_per_test):
                outcome = sample_outcome(context.regime, test, rng)
                samples.append(
                    InterventionSample(
                        context_id=context.context_id,
                        regime_index=context.regime_index,
                        features=context.features.copy(),
                        test_index=test_index,
                        outcome=outcome,
                    )
                )
    return samples


def generate_overlap_pairs(
    contexts: list[ContextSample],
    pairs_per_regime: int,
    rng: np.random.Generator,
) -> list[OverlapPair]:
    by_regime: dict[int, list[ContextSample]] = {}
    for context in contexts:
        by_regime.setdefault(context.regime_index, []).append(context)
    pairs: list[OverlapPair] = []
    for regime_index, members in by_regime.items():
        for _ in range(pairs_per_regime):
            a_idx, b_idx = rng.choice(len(members), size=2, replace=False)
            pairs.append(OverlapPair(a=members[int(a_idx)], b=members[int(b_idx)]))
    return pairs


def expected_outcome(regime: str, test: str) -> np.ndarray:
    """True outcome distribution vector for (regime, test)."""
    if test in DIRECT_ACTIONS:
        outcome = ACTION_MODEL[regime][test]
        return np.asarray([outcome.success, outcome.unsafe], dtype=float)
    return np.asarray([PROBE_LIKELIHOOD[test][regime], 0.0], dtype=float)


def regime_outcome_table() -> np.ndarray:
    """(N_REGIMES, N_TESTS, 2) table of true expected outcomes."""
    table = np.zeros((N_REGIMES, N_TESTS, 2), dtype=float)
    for ri, regime in enumerate(REGIMES):
        for ti, test in enumerate(TESTS):
            table[ri, ti] = expected_outcome(regime, test)
    return table
