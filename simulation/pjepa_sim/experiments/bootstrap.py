"""Paired bootstrap CI helper used by hypothesis experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapCI:
    mean: float
    low: float
    high: float
    n: int
    resamples: int
    confidence: float

    def contains_zero(self) -> bool:
        return self.low <= 0.0 <= self.high

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "mean": self.mean,
            "low": self.low,
            "high": self.high,
            "n": self.n,
            "resamples": self.resamples,
            "confidence": self.confidence,
            "contains_zero": self.contains_zero(),
        }


def paired_bootstrap_ci(
    deltas: list[float] | tuple[float, ...] | np.ndarray,
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapCI:
    """Percentile bootstrap CI on the mean of paired deltas.

    `deltas[i]` is the per-pair difference (e.g., active_score - entropy_score
    for seed i). Resamples with replacement at the pair level.
    """
    values = np.asarray(deltas, dtype=float)
    n = int(values.shape[0])
    if n < 2:
        raise ValueError("need at least two paired observations for a bootstrap CI")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(resamples, n))
    means = values[indices].mean(axis=1)
    low_q = (1.0 - confidence) / 2.0
    high_q = 1.0 - low_q
    low = float(np.quantile(means, low_q))
    high = float(np.quantile(means, high_q))
    return BootstrapCI(
        mean=float(values.mean()),
        low=low,
        high=high,
        n=n,
        resamples=resamples,
        confidence=confidence,
    )
