"""Run the pixel-observation continuous-control benchmark."""

from __future__ import annotations

import json

from pjepa_sim.paths import OUTPUT_DIR
from pjepa_sim.perception.continuous import run_pixel_continuous_benchmark


OUT = OUTPUT_DIR


def main() -> None:
    results = run_pixel_continuous_benchmark()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "pixel_continuous_benchmark.json"
    table_path = OUT / "pixel_continuous_benchmark.md"
    path.write_text(json.dumps(results, indent=2) + "\n")
    table_path.write_text(markdown_table(results))
    print_table(results)
    print()
    print(f"Wrote: {path}")
    print(f"Wrote: {table_path}")


def markdown_table(results: dict) -> str:
    lines = [
        "| Learner | Success | Unsafe | Probes | Score |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in results["learners"].items():
        lines.append(
            "| "
            f"`{name}` | "
            f"{metrics['success_rate']:.3f} | "
            f"{metrics['unsafe_failure_rate']:.3f} | "
            f"{metrics['mean_probes']:.3f} | "
            f"{metrics['risk_adjusted_score']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def print_table(results: dict) -> None:
    print("Pixel continuous-control benchmark")
    for name, metrics in results["learners"].items():
        print(
            f"  {name:>20s}  "
            f"success={metrics['success_rate']:.3f}  "
            f"unsafe={metrics['unsafe_failure_rate']:.3f}  "
            f"probes={metrics['mean_probes']:.3f}  "
            f"score={metrics['risk_adjusted_score']:.3f}"
        )


if __name__ == "__main__":
    main()
