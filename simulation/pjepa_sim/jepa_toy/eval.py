"""Evaluation of a trained JEPA-toy encoder.

Two metrics:
- cluster_purity: cluster the test contexts' latents with k-means
  (k = N_REGIMES), report majority-regime purity. Tests whether the
  latent organises around action consequence regimes despite the
  visual shift between train/test splits.
- action_score: assign each test context to its nearest train cluster
  centroid; choose the best utility-weighted action from per-cluster
  empirical sections (success - 2*unsafe). Report risk-adjusted score
  on the test contexts against the true regime outcomes.

For models with a viability head, also report viability-aware action
score: choose the action that maximises (success - 2 * predicted_unsafe).
"""

from __future__ import annotations

import numpy as np

from pjepa_sim.core.dishworld import ACTION_MODEL, DIRECT_ACTIONS, REGIMES
from pjepa_sim.jepa_toy.data import (
    CONTEXT_DIM,
    ContextSample,
    N_DIRECT_ACTIONS,
    N_REGIMES,
    N_TESTS,
    generate_contexts,
)
from pjepa_sim.jepa_toy.model import JEPAModel
from pjepa_sim.jepa_toy.training import make_action_one_hot
from pjepa_sim.representation.clustering import assign_nearest, best_kmeans, standardise


UNSAFE_WEIGHT = 2.0


def encode_contexts(model: JEPAModel, contexts: list[ContextSample]) -> np.ndarray:
    feats = np.asarray([c.features for c in contexts], dtype=float)
    return model.encode(feats)


def cluster_purity_from_assignments(contexts: list[ContextSample], assignments: np.ndarray) -> float:
    total = 0
    correct = 0
    for cluster in sorted({int(a) for a in assignments}):
        labels = [c.regime for c, a in zip(contexts, assignments) if int(a) == cluster]
        if not labels:
            continue
        counts = {label: labels.count(label) for label in set(labels)}
        correct += max(counts.values())
        total += len(labels)
    return float(correct / total) if total else 0.0


def empirical_section(contexts: list[ContextSample], cluster_idx: int, assignments: np.ndarray) -> dict[str, tuple[float, float]]:
    """Empirical (success, unsafe) per action for the cluster, computed
    by averaging the true expected outcomes of the cluster's members."""
    members = [c for c, a in zip(contexts, assignments) if int(a) == cluster_idx]
    if not members:
        # Fallback: uniform prior over regimes.
        global_section: dict[str, tuple[float, float]] = {}
        for action in DIRECT_ACTIONS:
            outcomes = [ACTION_MODEL[r][action] for r in REGIMES]
            global_section[action] = (
                float(np.mean([o.success for o in outcomes])),
                float(np.mean([o.unsafe for o in outcomes])),
            )
        return global_section
    section: dict[str, tuple[float, float]] = {}
    for action in DIRECT_ACTIONS:
        success = float(np.mean([ACTION_MODEL[c.regime][action].success for c in members]))
        unsafe = float(np.mean([ACTION_MODEL[c.regime][action].unsafe for c in members]))
        section[action] = (success, unsafe)
    return section


def best_action_from_section(section: dict[str, tuple[float, float]]) -> str:
    utilities = {a: section[a][0] - UNSAFE_WEIGHT * section[a][1] for a in DIRECT_ACTIONS}
    return max(utilities, key=utilities.get)


def evaluate_action_choice(
    model: JEPAModel,
    train_contexts: list[ContextSample],
    test_contexts: list[ContextSample],
    cluster_seed: int = 0,
) -> dict[str, float]:
    """k-means on standardised train latents; assign test latents to
    nearest train centroid; choose best action from per-cluster section."""
    train_latents = encode_contexts(model, train_contexts)
    test_latents = encode_contexts(model, test_contexts)
    train_norm, mean, std = standardise(train_latents)
    test_norm = (test_latents - mean) / std
    train_assignments, centers = best_kmeans(
        train_norm, k=N_REGIMES, seed=cluster_seed, restarts=24
    )
    test_assignments = assign_nearest(test_norm, centers)

    train_purity = cluster_purity_from_assignments(train_contexts, train_assignments)
    test_purity = cluster_purity_from_assignments(test_contexts, test_assignments)

    sections = {
        idx: empirical_section(train_contexts, idx, train_assignments)
        for idx in sorted({int(a) for a in train_assignments})
    }

    successes: list[float] = []
    unsafes: list[float] = []
    for ctx, cluster in zip(test_contexts, test_assignments):
        sec = sections[int(cluster)]
        action = best_action_from_section(sec)
        true_outcome = ACTION_MODEL[ctx.regime][action]
        successes.append(true_outcome.success)
        unsafes.append(true_outcome.unsafe)

    success_rate = float(np.mean(successes))
    unsafe_rate = float(np.mean(unsafes))
    return {
        "train_cluster_purity": train_purity,
        "test_cluster_purity": test_purity,
        "success_rate": success_rate,
        "unsafe_failure_rate": unsafe_rate,
        "risk_adjusted_score": success_rate - UNSAFE_WEIGHT * unsafe_rate,
    }


def evaluate_viability_action_choice(
    model: JEPAModel,
    test_contexts: list[ContextSample],
) -> dict[str, float]:
    """For models with a viability head: pick action that maximises
    (sigma * 1.0 - unsafe_weight * predicted_unsafe) where sigma is a
    placeholder constant success estimate. We use the empirical mean
    success per action from ACTION_MODEL averaged over regimes - i.e.,
    a 'no-info' baseline - as the success input. This isolates the
    viability head's contribution."""
    if model.viability_head is None:
        return {}
    test_feats = np.asarray([c.features for c in test_contexts], dtype=float)
    s_x = model.encode(test_feats)
    # Pre-compute prior success per action (averaged over regimes).
    prior_success = {
        action: float(np.mean([ACTION_MODEL[r][action].success for r in REGIMES]))
        for action in DIRECT_ACTIONS
    }
    n = len(test_contexts)
    successes: list[float] = []
    unsafes: list[float] = []
    for i, ctx in enumerate(test_contexts):
        utilities = []
        for ai, action in enumerate(DIRECT_ACTIONS):
            one_hot = make_action_one_hot(ai)
            pred_unsafe = float(model.predict_viability(s_x[i:i + 1], one_hot[None, :])[0, 0])
            utilities.append((prior_success[action] - UNSAFE_WEIGHT * pred_unsafe, action))
        utilities.sort(key=lambda item: (-item[0], item[1]))
        action = utilities[0][1]
        true_outcome = ACTION_MODEL[ctx.regime][action]
        successes.append(true_outcome.success)
        unsafes.append(true_outcome.unsafe)
    success_rate = float(np.mean(successes))
    unsafe_rate = float(np.mean(unsafes))
    return {
        "viability_success_rate": success_rate,
        "viability_unsafe_failure_rate": unsafe_rate,
        "viability_risk_adjusted_score": success_rate - UNSAFE_WEIGHT * unsafe_rate,
    }
