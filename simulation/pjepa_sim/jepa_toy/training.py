"""JEPA toy training loop with toggleable auxiliary losses."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pjepa_sim.core.dishworld import ACTION_MODEL, DIRECT_ACTIONS, REGIMES
from pjepa_sim.jepa_toy.data import (
    CONTEXT_DIM,
    ContextSample,
    InterventionSample,
    MaskedPair,
    N_DIRECT_ACTIONS,
    N_TESTS,
    OverlapPair,
    TESTS,
    generate_contexts,
    generate_interventions,
    generate_masked_pairs,
    generate_overlap_pairs,
    random_mask,
    regime_outcome_table,
)
from pjepa_sim.jepa_toy.losses import (
    accumulate_loss_grads,
    bisimulation_loss,
    empty_grads,
    intervention_loss,
    jepa_mask_loss,
    viability_loss,
)
from pjepa_sim.jepa_toy.model import (
    AdamState,
    JEPAModel,
    adam_step,
    ema_update,
)


@dataclass
class TrainingConfig:
    enable_intervention: bool = False
    enable_bisimulation: bool = False
    enable_viability: bool = False
    enable_active_masking: bool = False

    jepa_weight: float = 1.0
    intervention_weight: float = 1.0
    bisimulation_weight: float = 0.3
    viability_weight: float = 0.5

    latent_dim: int = 16
    hidden_dim: int = 32
    n_epochs: int = 600
    batch_size: int = 64
    learning_rate: float = 0.01
    ema_momentum: float = 0.99
    mask_fraction: float = 0.4
    active_candidate_count: int = 4

    contexts_per_regime: int = 96
    intervention_repeats_per_test: int = 1
    sensor_noise: float = 0.045
    overlap_pairs_per_regime: int = 64

    enabled_augmentations: tuple[str, ...] = field(default_factory=tuple)


def make_action_one_hot(test_index: int) -> np.ndarray:
    one_hot = np.zeros(N_TESTS, dtype=float)
    one_hot[test_index] = 1.0
    return one_hot


def regime_outcome_vector(regime: str) -> np.ndarray:
    """Concatenation over DIRECT_ACTIONS of (success, unsafe). Length 2 * N_DIRECT_ACTIONS."""
    parts: list[float] = []
    for action in DIRECT_ACTIONS:
        outcome = ACTION_MODEL[regime][action]
        parts.extend([outcome.success, outcome.unsafe])
    return np.asarray(parts, dtype=float)


def pairwise_target_distance(a_regime: str, b_regime: str) -> float:
    """L2 distance between regimes' true outcome vectors."""
    return float(np.linalg.norm(regime_outcome_vector(a_regime) - regime_outcome_vector(b_regime)))


def choose_active_mask(
    model: JEPAModel,
    full: np.ndarray,
    rng: np.random.Generator,
    n_candidates: int,
    mask_fraction: float,
) -> np.ndarray:
    """Hard-example mining mask sampler.

    For one context: sample n_candidates random masks, compute the
    current JEPA prediction error for each, return the highest-error
    mask. Approximates ensemble disagreement using the current model's
    own loss as an uncertainty proxy.
    """
    full = full.reshape(1, -1)
    s_y = model.target_encode(full)  # (1, latent)
    best_err = -np.inf
    best_mask = None
    for _ in range(n_candidates):
        mask = random_mask(rng, mask_fraction)
        view = full * mask
        s_x = model.encode(view)
        hat = model.predict_mask(s_x, mask.reshape(1, -1))
        err = float(np.mean((hat - s_y) ** 2))
        if err > best_err:
            best_err = err
            best_mask = mask
    assert best_mask is not None
    return best_mask


def select_masked_batch(
    model: JEPAModel,
    contexts: list[ContextSample],
    rng: np.random.Generator,
    config: TrainingConfig,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = rng.integers(0, len(contexts), size=batch_size)
    views: list[np.ndarray] = []
    fulls: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for idx in indices:
        ctx = contexts[int(idx)]
        if config.enable_active_masking:
            mask = choose_active_mask(
                model, ctx.features, rng,
                n_candidates=config.active_candidate_count,
                mask_fraction=config.mask_fraction,
            )
        else:
            mask = random_mask(rng, config.mask_fraction)
        views.append(ctx.features * mask)
        fulls.append(ctx.features.copy())
        masks.append(mask)
    return np.asarray(views), np.asarray(fulls), np.asarray(masks)


def select_intervention_batch(
    samples: list[InterventionSample],
    rng: np.random.Generator,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = rng.integers(0, len(samples), size=batch_size)
    features: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    outcomes: list[np.ndarray] = []
    for idx in indices:
        sample = samples[int(idx)]
        features.append(sample.features)
        actions.append(make_action_one_hot(sample.test_index))
        outcomes.append(sample.outcome)
    return np.asarray(features), np.asarray(actions), np.asarray(outcomes)


def select_overlap_batch(
    pairs: list[OverlapPair],
    cross_pairs: list[OverlapPair],
    rng: np.random.Generator,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mixture of within-regime (positive, target_dist ~ 0) and
    cross-regime (negative, target_dist > 0) pairs."""
    half = batch_size // 2
    pos_idx = rng.integers(0, len(pairs), size=half)
    neg_idx = rng.integers(0, len(cross_pairs), size=batch_size - half)
    a_feats: list[np.ndarray] = []
    b_feats: list[np.ndarray] = []
    targets: list[float] = []
    for idx in pos_idx:
        pair = pairs[int(idx)]
        a_feats.append(pair.a.features)
        b_feats.append(pair.b.features)
        targets.append(pairwise_target_distance(pair.a.regime, pair.b.regime))
    for idx in neg_idx:
        pair = cross_pairs[int(idx)]
        a_feats.append(pair.a.features)
        b_feats.append(pair.b.features)
        targets.append(pairwise_target_distance(pair.a.regime, pair.b.regime))
    return np.asarray(a_feats), np.asarray(b_feats), np.asarray(targets)


def select_viability_batch(
    contexts: list[ContextSample],
    rng: np.random.Generator,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = rng.integers(0, len(contexts), size=batch_size)
    action_indices = rng.integers(0, N_DIRECT_ACTIONS, size=batch_size)
    features: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    unsafe: list[float] = []
    for ctx_idx, act_idx in zip(indices, action_indices):
        ctx = contexts[int(ctx_idx)]
        features.append(ctx.features)
        actions.append(make_action_one_hot(int(act_idx)))
        action_name = DIRECT_ACTIONS[int(act_idx)]
        unsafe.append(ACTION_MODEL[ctx.regime][action_name].unsafe)
    return np.asarray(features), np.asarray(actions), np.asarray(unsafe)


def generate_cross_regime_pairs(
    contexts: list[ContextSample],
    pairs_per_regime: int,
    rng: np.random.Generator,
) -> list[OverlapPair]:
    by_regime: dict[int, list[ContextSample]] = {}
    for ctx in contexts:
        by_regime.setdefault(ctx.regime_index, []).append(ctx)
    pairs: list[OverlapPair] = []
    regime_ids = list(by_regime.keys())
    for i_idx, ri in enumerate(regime_ids):
        for rj in regime_ids[i_idx + 1:]:
            members_i = by_regime[ri]
            members_j = by_regime[rj]
            for _ in range(pairs_per_regime):
                a = members_i[int(rng.integers(0, len(members_i)))]
                b = members_j[int(rng.integers(0, len(members_j)))]
                pairs.append(OverlapPair(a=a, b=b))
    return pairs


@dataclass
class TrainingResult:
    model: JEPAModel
    final_losses: dict[str, float]
    loss_history: list[dict[str, float]]


def train_jepa_toy(
    config: TrainingConfig,
    seed: int,
) -> TrainingResult:
    rng = np.random.default_rng(seed)
    train_contexts = generate_contexts(
        "train", config.contexts_per_regime, config.sensor_noise, rng
    )
    intervention_samples: list[InterventionSample] = []
    if config.enable_intervention:
        intervention_samples = generate_interventions(
            train_contexts, config.intervention_repeats_per_test, rng
        )
    overlap_pairs: list[OverlapPair] = []
    cross_pairs: list[OverlapPair] = []
    if config.enable_bisimulation:
        overlap_pairs = generate_overlap_pairs(
            train_contexts, config.overlap_pairs_per_regime, rng
        )
        cross_pairs = generate_cross_regime_pairs(
            train_contexts, config.overlap_pairs_per_regime, rng
        )

    model = JEPAModel.create(
        input_dim=CONTEXT_DIM,
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
        n_tests=N_TESTS,
        with_outcome=config.enable_intervention,
        with_viability=config.enable_viability,
        seed=seed + 1,
    )
    adam_states = model.adam_states()

    loss_history: list[dict[str, float]] = []
    final_losses: dict[str, float] = {}

    for epoch in range(config.n_epochs):
        # ----- accumulate gradients across all enabled losses -----
        grads = empty_grads(model)
        epoch_losses: dict[str, float] = {}

        # Base JEPA mask-fill loss (always on)
        views, fulls, masks = select_masked_batch(
            model, train_contexts, rng, config, config.batch_size
        )
        l_jepa, g_jepa = jepa_mask_loss(model, views, fulls, masks)
        accumulate_loss_grads(grads, g_jepa, config.jepa_weight)
        epoch_losses["jepa"] = l_jepa

        if config.enable_intervention:
            feats, actions, outcomes = select_intervention_batch(
                intervention_samples, rng, config.batch_size
            )
            l_int, g_int = intervention_loss(model, feats, actions, outcomes)
            accumulate_loss_grads(grads, g_int, config.intervention_weight)
            epoch_losses["intervention"] = l_int

        if config.enable_bisimulation:
            a_feats, b_feats, targets = select_overlap_batch(
                overlap_pairs, cross_pairs, rng, config.batch_size
            )
            l_bis, g_bis = bisimulation_loss(model, a_feats, b_feats, targets)
            accumulate_loss_grads(grads, g_bis, config.bisimulation_weight)
            epoch_losses["bisimulation"] = l_bis

        if config.enable_viability:
            feats, actions, unsafe = select_viability_batch(
                train_contexts, rng, config.batch_size
            )
            l_via, g_via = viability_loss(model, feats, actions, unsafe)
            accumulate_loss_grads(grads, g_via, config.viability_weight)
            epoch_losses["viability"] = l_via

        # ----- Adam step on each parameter set with non-zero grads -----
        adam_step(model.encoder, grads["encoder"], adam_states["encoder"],
                  lr=config.learning_rate)
        adam_step(model.mask_predictor, grads["mask_predictor"],
                  adam_states["mask_predictor"], lr=config.learning_rate)
        if config.enable_intervention:
            adam_step(model.outcome_predictor, grads["outcome_predictor"],
                      adam_states["outcome_predictor"], lr=config.learning_rate)
        if config.enable_viability:
            adam_step(model.viability_head, grads["viability_head"],
                      adam_states["viability_head"], lr=config.learning_rate)

        # ----- EMA update of target encoder -----
        ema_update(model.target_encoder, model.encoder, momentum=config.ema_momentum)

        if (epoch + 1) % max(1, config.n_epochs // 12) == 0:
            loss_history.append({"epoch": epoch + 1, **epoch_losses})
        final_losses = epoch_losses

    return TrainingResult(model=model, final_losses=final_losses, loss_history=loss_history)
