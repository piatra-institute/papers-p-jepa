"""Run the P-JEPA online cover-construction benchmark."""

from __future__ import annotations

import json

from pjepa_sim.paths import OUTPUT_DIR
from pjepa_sim.representation.online import run_online_cover_benchmark


OUT = OUTPUT_DIR


def main() -> None:
    results = run_online_cover_benchmark()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "online_cover_benchmark.json"
    table_path = OUT / "online_cover_benchmark.md"
    path.write_text(json.dumps(results, indent=2) + "\n")
    table_path.write_text(markdown_table(results))
    print_table(results)
    print()
    print(f"Wrote: {path}")
    print(f"Wrote: {table_path}")


def markdown_table(results: dict) -> str:
    lines = [
        "| Learner | Success | Unsafe | Score | Purity | Clusters |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in results["learners"].items():
        lines.append(
            "| "
            f"`{name}` | "
            f"{metrics['success_rate']:.3f} | "
            f"{metrics['unsafe_failure_rate']:.3f} | "
            f"{metrics['risk_adjusted_score']:.3f} | "
            f"{metrics['cluster_purity']:.3f} | "
            f"{metrics['n_clusters']} |"
        )
    return "\n".join(lines) + "\n"


def print_table(results: dict) -> None:
    print("Online cover-construction benchmark")
    for name, metrics in results["learners"].items():
        print(
            f"  {name:>24s}  "
            f"success={metrics['success_rate']:.3f}  "
            f"unsafe={metrics['unsafe_failure_rate']:.3f}  "
            f"score={metrics['risk_adjusted_score']:.3f}  "
            f"purity={metrics['cluster_purity']:.3f}  "
            f"clusters={metrics['n_clusters']}"
        )


if __name__ == "__main__":
    main()
