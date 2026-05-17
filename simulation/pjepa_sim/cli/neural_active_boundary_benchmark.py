"""Run the learned neural active-probing boundary-condition benchmark."""

from __future__ import annotations

import json

from pjepa_sim.paths import OUTPUT_DIR
from pjepa_sim.representation.neural_active import run_neural_active_boundary_benchmark


OUT = OUTPUT_DIR


def main() -> None:
    results = run_neural_active_boundary_benchmark()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "neural_active_boundary_benchmark.json"
    table_path = OUT / "neural_active_boundary_benchmark.md"
    path.write_text(json.dumps(results, indent=2) + "\n")
    table_path.write_text(markdown_table(results))
    print_table(results)
    print()
    print(f"Wrote: {path}")
    print(f"Wrote: {table_path}")


def markdown_table(results: dict) -> str:
    lines = [
        "| Case | No Probe | Entropy | Active | Unsafe Reduction | Active Probes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, case in results["cases"].items():
        learners = case["learners"]
        no_probe = learners["learned_no_probe"]
        entropy = learners["learned_entropy_probe"]
        active = learners["learned_active_probe"]
        lines.append(
            "| "
            f"`{name}` | "
            f"{no_probe['risk_adjusted_score']:.3f} | "
            f"{entropy['risk_adjusted_score']:.3f} | "
            f"{active['risk_adjusted_score']:.3f} | "
            f"{no_probe['unsafe_failure_rate'] - active['unsafe_failure_rate']:.3f} | "
            f"{active['mean_probes']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def print_table(results: dict) -> None:
    print("Neural active-probe boundary benchmark")
    for name, case in results["cases"].items():
        learners = case["learners"]
        no_probe = learners["learned_no_probe"]
        entropy = learners["learned_entropy_probe"]
        active = learners["learned_active_probe"]
        print(
            f"  {name:>24s}  "
            f"no_probe={no_probe['risk_adjusted_score']:.3f}  "
            f"entropy={entropy['risk_adjusted_score']:.3f}  "
            f"active={active['risk_adjusted_score']:.3f}  "
            f"unsafe_reduction={no_probe['unsafe_failure_rate'] - active['unsafe_failure_rate']:.3f}  "
            f"active_probes={active['mean_probes']:.3f}"
        )


if __name__ == "__main__":
    main()
