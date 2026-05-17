"""Run the local video-representation surrogate benchmark."""

from __future__ import annotations

import json

from pjepa_sim.paths import OUTPUT_DIR
from pjepa_sim.perception.video_representation import run_video_representation_benchmark


def main() -> None:
    results = run_video_representation_benchmark()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "video_representation_benchmark.json"
    md_path = OUTPUT_DIR / "video_representation_benchmark.md"
    json_path.write_text(json.dumps(results, indent=2) + "\n")
    md_path.write_text(markdown_table(results))
    print_table(results)
    print()
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


def markdown_table(results: dict) -> str:
    lines = [
        "| Learner | Success | Unsafe | Score | Purity | Passive MAE | Action MAE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in results["learners"].items():
        lines.append(
            "| "
            f"`{name}` | "
            f"{metrics['success_rate']:.3f} | "
            f"{metrics['unsafe_failure_rate']:.3f} | "
            f"{metrics['risk_adjusted_score']:.3f} | "
            f"{metrics['cluster_purity']:.3f} | "
            f"{_fmt(metrics['passive_prediction_mae'])} | "
            f"{_fmt(metrics['action_feature_mae'])} |"
        )
    return "\n".join(lines) + "\n"


def print_table(results: dict) -> None:
    print("Video representation surrogate benchmark")
    for name, metrics in results["learners"].items():
        print(
            f"  {name:>28s}  "
            f"success={metrics['success_rate']:.3f}  "
            f"unsafe={metrics['unsafe_failure_rate']:.3f}  "
            f"score={metrics['risk_adjusted_score']:.3f}  "
            f"purity={metrics['cluster_purity']:.3f}"
        )


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    main()

