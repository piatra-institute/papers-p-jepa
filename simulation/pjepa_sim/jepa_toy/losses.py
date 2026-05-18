"""JEPA toy losses with manual backprop.

Each loss returns (scalar value, dict of parameter-set name -> MLPParams
of gradient contributions). The training loop accumulates them with
per-loss weights and applies Adam to each parameter set.

Losses implemented:
- jepa_mask_loss: standard JEPA mask-fill loss.
- intervention_loss: predict (success, unsafe) for (context, test).
- bisimulation_loss: latent distance matches outcome-distribution
  distance between paired contexts.
- viability_loss: predict unsafe rate from (latent, action).

Active masking is implemented as a mask sampler (hard-example mining),
not as a separate loss term — see training.choose_active_mask.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from pjepa_sim.jepa_toy.model import (
    JEPAModel,
    MLPParams,
    mlp_backward,
    mlp_forward,
    zero_grads_like,
)


def jepa_mask_loss(
    model: JEPAModel,
    views: np.ndarray,
    fulls: np.ndarray,
    masks: np.ndarray,
) -> tuple[float, dict[str, MLPParams]]:
    """L = mean ||predict_mask(encoder(view), mask) - sg(target_encoder(full))||^2

    views, fulls: (B, input_dim). masks: (B, input_dim).
    Returns grads for {encoder, mask_predictor}.
    """
    enc_cache = mlp_forward(model.encoder, views)
    s_x = enc_cache["logits"]
    s_y = mlp_forward(model.target_encoder, fulls)["logits"]  # no grad

    pred_input = np.concatenate([s_x, masks], axis=-1)
    pred_cache = mlp_forward(model.mask_predictor, pred_input)
    hat_s_y = pred_cache["logits"]

    diff = hat_s_y - s_y
    loss_value = float(np.mean(diff * diff))
    dlogits = (2.0 / diff.size) * diff

    dinput_pred, predictor_grads = mlp_backward(model.mask_predictor, pred_cache, dlogits)
    # dinput_pred has shape (B, latent_dim + input_dim). Split.
    d_s_x = dinput_pred[:, : model.latent_dim]
    # mask contribution to predictor input has no learnable upstream params.

    _, encoder_grads = mlp_backward(model.encoder, enc_cache, d_s_x)
    return loss_value, {"encoder": encoder_grads, "mask_predictor": predictor_grads}


def intervention_loss(
    model: JEPAModel,
    features: np.ndarray,
    action_one_hots: np.ndarray,
    outcomes: np.ndarray,
) -> tuple[float, dict[str, MLPParams]]:
    """L = mean ||outcome_predictor(encoder(x), action) - outcome||^2

    features: (B, input_dim). action_one_hots: (B, n_tests). outcomes: (B, 2).
    Returns grads for {encoder, outcome_predictor}.
    """
    assert model.outcome_predictor is not None, "intervention_loss requires outcome head"

    enc_cache = mlp_forward(model.encoder, features)
    s_x = enc_cache["logits"]

    pred_input = np.concatenate([s_x, action_one_hots], axis=-1)
    pred_cache = mlp_forward(model.outcome_predictor, pred_input)
    hat_outcome = pred_cache["logits"]

    diff = hat_outcome - outcomes
    loss_value = float(np.mean(diff * diff))
    dlogits = (2.0 / diff.size) * diff

    dinput_pred, predictor_grads = mlp_backward(model.outcome_predictor, pred_cache, dlogits)
    d_s_x = dinput_pred[:, : model.latent_dim]
    _, encoder_grads = mlp_backward(model.encoder, enc_cache, d_s_x)
    return loss_value, {"encoder": encoder_grads, "outcome_predictor": predictor_grads}


def bisimulation_loss(
    model: JEPAModel,
    features_a: np.ndarray,
    features_b: np.ndarray,
    target_distances: np.ndarray,
) -> tuple[float, dict[str, MLPParams]]:
    """L = mean (||encoder(a) - encoder(b)|| - target_distance)^2

    Returns grads for {encoder} (encoder is used for both branches; the
    gradient is the sum of the two branches' contributions).
    """
    enc_a = mlp_forward(model.encoder, features_a)
    enc_b = mlp_forward(model.encoder, features_b)
    s_a = enc_a["logits"]
    s_b = enc_b["logits"]
    delta = s_a - s_b
    norms = np.linalg.norm(delta, axis=-1) + 1e-8  # (B,)
    residual = norms - target_distances  # (B,)
    loss_value = float(np.mean(residual * residual))

    # dL/d_norms = 2 * residual / B
    d_norms = (2.0 / residual.size) * residual  # (B,)
    # d_norms / d_delta = delta / ||delta||
    d_delta = (d_norms / norms)[:, None] * delta  # (B, latent_dim)
    d_s_a = d_delta
    d_s_b = -d_delta

    _, grads_a = mlp_backward(model.encoder, enc_a, d_s_a)
    _, grads_b = mlp_backward(model.encoder, enc_b, d_s_b)

    # Sum the two encoder-branch gradients (same parameters).
    grads_total = MLPParams(
        w1=grads_a.w1 + grads_b.w1,
        b1=grads_a.b1 + grads_b.b1,
        w2=grads_a.w2 + grads_b.w2,
        b2=grads_a.b2 + grads_b.b2,
    )
    return loss_value, {"encoder": grads_total}


def viability_loss(
    model: JEPAModel,
    features: np.ndarray,
    action_one_hots: np.ndarray,
    true_unsafe: np.ndarray,
) -> tuple[float, dict[str, MLPParams]]:
    """L = mean (sigmoid(viability_head(encoder(x), action)) - true_unsafe)^2

    Returns grads for {encoder, viability_head}.
    """
    assert model.viability_head is not None, "viability_loss requires viability head"

    enc_cache = mlp_forward(model.encoder, features)
    s_x = enc_cache["logits"]
    pred_input = np.concatenate([s_x, action_one_hots], axis=-1)
    pred_cache = mlp_forward(model.viability_head, pred_input)
    logits = pred_cache["logits"]
    sigmoid = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    diff = sigmoid - true_unsafe[:, None]
    loss_value = float(np.mean(diff * diff))
    # dL/dlogits = 2 * diff / B * sigmoid * (1 - sigmoid)
    dlogits = (2.0 / diff.size) * diff * sigmoid * (1.0 - sigmoid)
    dinput_pred, head_grads = mlp_backward(model.viability_head, pred_cache, dlogits)
    d_s_x = dinput_pred[:, : model.latent_dim]
    _, encoder_grads = mlp_backward(model.encoder, enc_cache, d_s_x)
    return loss_value, {"encoder": encoder_grads, "viability_head": head_grads}


def empty_grads(model: JEPAModel) -> dict[str, MLPParams]:
    grads: dict[str, MLPParams] = {
        "encoder": zero_grads_like(model.encoder),
        "mask_predictor": zero_grads_like(model.mask_predictor),
    }
    if model.outcome_predictor is not None:
        grads["outcome_predictor"] = zero_grads_like(model.outcome_predictor)
    if model.viability_head is not None:
        grads["viability_head"] = zero_grads_like(model.viability_head)
    return grads


def accumulate_loss_grads(
    accumulator: dict[str, MLPParams],
    contribution: dict[str, MLPParams],
    weight: float,
) -> None:
    for key, grads in contribution.items():
        target = accumulator[key]
        target.w1 += weight * grads.w1
        target.b1 += weight * grads.b1
        target.w2 += weight * grads.w2
        target.b2 += weight * grads.b2
