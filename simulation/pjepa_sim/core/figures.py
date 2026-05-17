"""Figure generation for the P-JEPA simulation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pjepa_sim.core.dishworld import REGIMES


def _agent_names(results: dict) -> list[str]:
    return list(results["agents"].keys())


def plot_agent_bars(results: dict, savepath: str) -> None:
    Path(savepath).parent.mkdir(parents=True, exist_ok=True)
    names = _agent_names(results)
    success = [results["agents"][n]["success_rate"] for n in names]
    unsafe = [results["agents"][n]["unsafe_failure_rate"] for n in names]
    x = np.arange(len(names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - width / 2, success, width, label="success", color="#2f7f62")
    ax.bar(x + width / 2, unsafe, width, label="unsafe failure", color="#b85c50")
    ax.set_ylim(0, 1)
    ax.set_ylabel("expected rate")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_title("Hidden-regime manipulation outcomes")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(savepath, dpi=180)
    plt.close(fig)


def plot_obstruction(results: dict, savepath: str) -> None:
    Path(savepath).parent.mkdir(parents=True, exist_ok=True)
    sheaf = results["agents"]["sheaf_probe"]
    before = [results["obstruction"]["prior"] for _ in REGIMES]
    after = [sheaf["by_regime"][r]["obstruction_at_action"] for r in REGIMES]
    x = np.arange(len(REGIMES))
    width = 0.38

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(x - width / 2, before, width, label="before probes", color="#5b6f95")
    ax.bar(x + width / 2, after, width, label="at action", color="#d59f45")
    ax.set_ylabel(r"$\|d\sigma\|^2$")
    ax.set_xticks(x)
    ax.set_xticklabels(REGIMES)
    ax.set_title("Obstruction reduction by hidden regime")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(savepath, dpi=180)
    plt.close(fig)


def plot_transfer(results: dict, savepath: str) -> None:
    Path(savepath).parent.mkdir(parents=True, exist_ok=True)
    names = _agent_names(results)
    data = np.array(
        [
            [results["agents"][name]["by_regime"][regime]["success_rate"] for regime in REGIMES]
            for name in names
        ]
    )

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    im = ax.imshow(data, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(REGIMES)))
    ax.set_xticklabels(REGIMES)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels([n.replace("_", " ") for n in names])
    ax.set_title("Success rate by hidden regime")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="white")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("expected success")
    fig.tight_layout()
    fig.savefig(savepath, dpi=180)
    plt.close(fig)
