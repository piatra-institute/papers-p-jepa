"""NumPy MLPs for the JEPA toy.

Components:
- Encoder: context features -> latent.
- Target encoder: EMA copy of encoder, no gradients.
- Mask predictor: (latent, mask_vector) -> latent. Predicts the target
  encoder's latent of the full context.
- Outcome predictor: (latent, action_one_hot) -> R^2. Predicts
  (success, unsafe) of running a test against the context (intervention
  loss).
- Viability head: (latent, action_one_hot) -> R^1 in (0, 1) via sigmoid.
  Predicts unsafe rate.

All heads are 2-layer MLPs. Adam optimizer is hand-rolled. Manual
backprop because the toy is small enough to make explicit gradients
worth the readability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def he_init(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    return rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))


@dataclass
class MLPParams:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    @classmethod
    def create(
        cls, input_dim: int, hidden_dim: int, output_dim: int, rng: np.random.Generator
    ) -> "MLPParams":
        return cls(
            w1=he_init(rng, input_dim, hidden_dim),
            b1=np.zeros(hidden_dim),
            w2=he_init(rng, hidden_dim, output_dim),
            b2=np.zeros(output_dim),
        )

    def copy(self) -> "MLPParams":
        return MLPParams(
            w1=self.w1.copy(), b1=self.b1.copy(),
            w2=self.w2.copy(), b2=self.b2.copy(),
        )

    def as_tuple(self) -> tuple[np.ndarray, ...]:
        return (self.w1, self.b1, self.w2, self.b2)

    def assign_from_tuple(self, values: tuple[np.ndarray, ...]) -> None:
        self.w1, self.b1, self.w2, self.b2 = values


def mlp_forward(params: MLPParams, x: np.ndarray) -> dict:
    z1 = x @ params.w1 + params.b1
    h = np.tanh(z1)
    logits = h @ params.w2 + params.b2
    return {"x": x, "z1": z1, "h": h, "logits": logits}


def mlp_backward(params: MLPParams, cache: dict, dlogits: np.ndarray) -> tuple[np.ndarray, MLPParams]:
    """Return (gradient wrt input, grads as MLPParams)."""
    h = cache["h"]
    x = cache["x"]
    dw2 = h.T @ dlogits
    db2 = dlogits.sum(axis=0)
    dh = dlogits @ params.w2.T
    dz1 = dh * (1.0 - h * h)
    dw1 = x.T @ dz1
    db1 = dz1.sum(axis=0)
    dx = dz1 @ params.w1.T
    return dx, MLPParams(w1=dw1, b1=db1, w2=dw2, b2=db2)


def zero_grads_like(params: MLPParams) -> MLPParams:
    return MLPParams(
        w1=np.zeros_like(params.w1),
        b1=np.zeros_like(params.b1),
        w2=np.zeros_like(params.w2),
        b2=np.zeros_like(params.b2),
    )


def accumulate_grads(target: MLPParams, contribution: MLPParams, weight: float = 1.0) -> None:
    target.w1 += weight * contribution.w1
    target.b1 += weight * contribution.b1
    target.w2 += weight * contribution.w2
    target.b2 += weight * contribution.b2


class AdamState:
    def __init__(self, params: MLPParams):
        self.m: list[np.ndarray] = [np.zeros_like(p) for p in params.as_tuple()]
        self.v: list[np.ndarray] = [np.zeros_like(p) for p in params.as_tuple()]
        self.t = 0


def adam_step(
    params: MLPParams,
    grads: MLPParams,
    state: AdamState,
    lr: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> None:
    state.t += 1
    param_arrays = list(params.as_tuple())
    grad_arrays = list(grads.as_tuple())
    updated: list[np.ndarray] = []
    for i, (p, g) in enumerate(zip(param_arrays, grad_arrays)):
        state.m[i] = beta1 * state.m[i] + (1.0 - beta1) * g
        state.v[i] = beta2 * state.v[i] + (1.0 - beta2) * (g * g)
        m_hat = state.m[i] / (1.0 - beta1 ** state.t)
        v_hat = state.v[i] / (1.0 - beta2 ** state.t)
        updated.append(p - lr * m_hat / (np.sqrt(v_hat) + eps))
    params.assign_from_tuple(tuple(updated))


def ema_update(target: MLPParams, online: MLPParams, momentum: float = 0.996) -> None:
    target_arrays = list(target.as_tuple())
    online_arrays = list(online.as_tuple())
    updated = [momentum * t + (1.0 - momentum) * o for t, o in zip(target_arrays, online_arrays)]
    target.assign_from_tuple(tuple(updated))


@dataclass
class JEPAModel:
    encoder: MLPParams
    target_encoder: MLPParams
    mask_predictor: MLPParams
    outcome_predictor: MLPParams | None
    viability_head: MLPParams | None
    input_dim: int
    latent_dim: int
    hidden_dim: int
    n_tests: int

    @classmethod
    def create(
        cls,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        n_tests: int,
        with_outcome: bool,
        with_viability: bool,
        seed: int,
    ) -> "JEPAModel":
        rng = np.random.default_rng(seed)
        encoder = MLPParams.create(input_dim, hidden_dim, latent_dim, rng)
        target_encoder = encoder.copy()
        # Mask predictor takes latent + mask vector (input_dim) and predicts latent.
        mask_predictor = MLPParams.create(latent_dim + input_dim, hidden_dim, latent_dim, rng)
        outcome_predictor = None
        if with_outcome:
            outcome_predictor = MLPParams.create(latent_dim + n_tests, hidden_dim, 2, rng)
        viability_head = None
        if with_viability:
            viability_head = MLPParams.create(latent_dim + n_tests, hidden_dim, 1, rng)
        return cls(
            encoder=encoder,
            target_encoder=target_encoder,
            mask_predictor=mask_predictor,
            outcome_predictor=outcome_predictor,
            viability_head=viability_head,
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_tests=n_tests,
        )

    # Convenience accessors -------------------------------------------------

    def encode(self, x: np.ndarray) -> np.ndarray:
        return mlp_forward(self.encoder, x)["logits"]

    def target_encode(self, x: np.ndarray) -> np.ndarray:
        return mlp_forward(self.target_encoder, x)["logits"]

    def predict_mask(self, s_x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        inp = np.concatenate([s_x, mask], axis=-1)
        return mlp_forward(self.mask_predictor, inp)["logits"]

    def predict_outcome(self, s_x: np.ndarray, action_one_hot: np.ndarray) -> np.ndarray:
        assert self.outcome_predictor is not None
        inp = np.concatenate([s_x, action_one_hot], axis=-1)
        return mlp_forward(self.outcome_predictor, inp)["logits"]

    def predict_viability(self, s_x: np.ndarray, action_one_hot: np.ndarray) -> np.ndarray:
        assert self.viability_head is not None
        inp = np.concatenate([s_x, action_one_hot], axis=-1)
        logits = mlp_forward(self.viability_head, inp)["logits"]
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))

    # Adam state factory ----------------------------------------------------

    def adam_states(self) -> dict[str, AdamState]:
        states: dict[str, AdamState] = {
            "encoder": AdamState(self.encoder),
            "mask_predictor": AdamState(self.mask_predictor),
        }
        if self.outcome_predictor is not None:
            states["outcome_predictor"] = AdamState(self.outcome_predictor)
        if self.viability_head is not None:
            states["viability_head"] = AdamState(self.viability_head)
        return states
