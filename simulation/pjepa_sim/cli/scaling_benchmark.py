"""Run the synthetic representation-scaling benchmark."""

from __future__ import annotations

import json

from pjepa_sim.paths import OUTPUT_DIR
from pjepa_sim.representation.scaling import run_scaling_benchmark


OUT = OUTPUT_DIR


def main() -> None:
    results = run_scaling_benchmark()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "scaling_benchmark.json"
    table_path = OUT / "scaling_benchmark.md"
    path.write_text(json.dumps(results, indent=2) + "\n")
    table_path.write_text(markdown_table(results))
    print_table(results)
    print()
    print(f"Wrote: {path}")
    print(f"Wrote: {table_path}")


def markdown_table(results: dict) -> str:
    lines = [
        "| Regimes | Learner | Success | Unsafe | Score | Purity | Clusters |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for n_regimes, case in results["cases"].items():
        for name, metrics in case["learners"].items():
            lines.append(
                "| "
                f"{n_regimes} | "
                f"`{name}` | "
                f"{metrics['success_rate']:.3f} | "
                f"{metrics['unsafe_failure_rate']:.3f} | "
                f"{metrics['risk_adjusted_score']:.3f} | "
                f"{metrics['cluster_purity']:.3f} | "
                f"{metrics['n_clusters']} |"
            )
    return "\n".join(lines) + "\n"


def print_table(results: dict) -> None:
    print("Synthetic representation-scaling benchmark")
    for n_regimes, case in results["cases"].items():
        print(f"  regimes={n_regimes}")
        for name, metrics in case["learners"].items():
            print(
                f"    {name:>28s}  "
                f"success={metrics['success_rate']:.3f}  "
                f"unsafe={metrics['unsafe_failure_rate']:.3f}  "
                f"score={metrics['risk_adjusted_score']:.3f}  "
                f"purity={metrics['cluster_purity']:.3f}  "
                f"clusters={metrics['n_clusters']}"
            )


if __name__ == "__main__":
    main()
