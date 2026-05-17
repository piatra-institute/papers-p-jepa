"""Run P-JEPA benchmark suites.

Examples:

    uv run python -m pjepa_sim.cli.benchmark --suite all --agents all
    uv run python -m pjepa_sim.cli.benchmark --suite noisy_probe_v0 --agents sheaf_probe psr_only
"""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from pjepa_sim.benchmark.suites import available_suites, evaluate_suite, load_spec
from pjepa_sim.paths import FIGURES_DIR, OUTPUT_DIR


OUT = OUTPUT_DIR
FIGURES = FIGURES_DIR
DEFAULT_AGENTS = [
    "visual_policy",
    "model_based_prior",
    "viability_prior",
    "jepa_prior",
    "psr_only",
    "active_psr_probe",
    "mixture_no_glue",
    "entropy_probe",
    "sheaf_probe",
    "p_jepa_stack",
    "oracle_hidden_regime",
]


def main() -> None:
    args = parse_args()
    suites = available_suites() if args.suite == ["all"] else args.suite
    agents = None if args.agents == ["all"] else args.agents

    results = {
        "benchmark": "p_jepa_hidden_regime",
        "suites": {},
    }
    for suite_name in suites:
        spec = load_spec(suite_name)
        results["suites"][suite_name] = evaluate_suite(spec, agents)

    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    write_json(results, OUT / "benchmark_results.json")
    write_markdown_table(results, OUT / "benchmark_table.md")
    plot_summary(results, FIGURES / "benchmark_summary.png")
    print_table(results)
    print()
    print(f"Wrote: {OUT / 'benchmark_results.json'}")
    print(f"Wrote: {OUT / 'benchmark_table.md'}")
    print(f"Wrote: {FIGURES / 'benchmark_summary.png'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        nargs="+",
        default=["all"],
        help="Suite names or 'all'.",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["all"],
        help="Agent names or 'all'.",
    )
    return parser.parse_args()


def write_json(results: dict, path: Path) -> None:
    with path.open("w") as f:
        json.dump(results, f, indent=2)


def write_markdown_table(results: dict, path: Path) -> None:
    lines = [
        "| Suite | Agent | Success | Unsafe | Probes | Obstruction at action | Risk-adjusted |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for suite_name, suite in results["suites"].items():
        for agent_name, metrics in suite["agents"].items():
            lines.append(
                "| "
                f"{suite_name} | `{agent_name}` | "
                f"{metrics['success_rate']:.3f} | "
                f"{metrics['unsafe_failure_rate']:.3f} | "
                f"{metrics['mean_probes']:.3f} | "
                f"{metrics['mean_obstruction_at_action']:.3f} | "
                f"{metrics['risk_adjusted_score']:.3f} |"
            )
    path.write_text("\n".join(lines) + "\n")


def print_table(results: dict) -> None:
    print("P-JEPA benchmark results")
    for suite_name, suite in results["suites"].items():
        print()
        print(suite_name)
        for agent_name, metrics in suite["agents"].items():
            print(
                f"  {agent_name:>20s}  "
                f"success={metrics['success_rate']:.3f}  "
                f"unsafe={metrics['unsafe_failure_rate']:.3f}  "
                f"probes={metrics['mean_probes']:.3f}  "
                f"O={metrics['mean_obstruction_at_action']:.3f}  "
                f"score={metrics['risk_adjusted_score']:.3f}"
            )


def plot_summary(results: dict, savepath: Path) -> None:
    suite_names = list(results["suites"].keys())
    agent_names = list(next(iter(results["suites"].values()))["agents"].keys())
    metrics = [
        ("success_rate", "Success", 0.0, 1.0),
        ("unsafe_failure_rate", "Unsafe failure", 0.0, 1.0),
        ("mean_obstruction_at_action", "Obstruction at action", 0.0, None),
        ("risk_adjusted_score", "Risk-adjusted score", None, None),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    x = np.arange(len(suite_names))
    width = 0.80 / len(agent_names)

    for ax, (metric_key, title, ymin, ymax) in zip(axes.ravel(), metrics):
        for i, agent_name in enumerate(agent_names):
            values = [
                results["suites"][suite]["agents"][agent_name][metric_key]
                for suite in suite_names
            ]
            offset = (i - (len(agent_names) - 1) / 2) * width
            ax.bar(x + offset, values, width, label=agent_name.replace("_", " "))
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([name.replace("_", "\n") for name in suite_names], fontsize=8)
        if ymin is not None or ymax is not None:
            ax.set_ylim(ymin, ymax)
        ax.grid(axis="y", alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(savepath, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
