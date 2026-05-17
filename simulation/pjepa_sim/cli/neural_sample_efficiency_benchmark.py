"""Run the neural intervention sample-efficiency benchmark."""

from __future__ import annotations

import json

from pjepa_sim.paths import OUTPUT_DIR
from pjepa_sim.representation.neural import run_neural_sample_efficiency_benchmark


OUT = OUTPUT_DIR


def main() -> None:
    results = run_neural_sample_efficiency_benchmark()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "neural_sample_efficiency_benchmark.json"
    table_path = OUT / "neural_sample_efficiency_benchmark.md"
    path.write_text(json.dumps(results, indent=2) + "\n")
    table_path.write_text(markdown_table(results))
    print_table(results)
    print()
    print(f"Wrote: {path}")
    print(f"Wrote: {table_path}")


def markdown_table(results: dict) -> str:
    lines = [
        "| Repeats | Prior | Appearance | Neural P | Engineered Ref. | Purity | Pred. MAE |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for repeats, case in results["cases"].items():
        learners = case["learners"]
        neural = learners["neural_p_representation"]
        lines.append(
            "| "
            f"{repeats} | "
            f"{learners['prior_average']['risk_adjusted_score']:.3f} | "
            f"{learners['appearance_only_encoder']['risk_adjusted_score']:.3f} | "
            f"{neural['risk_adjusted_score']:.3f} | "
            f"{learners['engineered_fingerprint_reference']['risk_adjusted_score']:.3f} | "
            f"{neural['cluster_purity']:.3f} | "
            f"{neural['mean_prediction_error']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def print_table(results: dict) -> None:
    print("Neural intervention sample-efficiency benchmark")
    for repeats, case in results["cases"].items():
        learners = case["learners"]
        neural = learners["neural_p_representation"]
        print(
            f"  repeats={int(repeats):>2d}  "
            f"prior={learners['prior_average']['risk_adjusted_score']:.3f}  "
            f"appearance={learners['appearance_only_encoder']['risk_adjusted_score']:.3f}  "
            f"neural={neural['risk_adjusted_score']:.3f}  "
            f"engineered={learners['engineered_fingerprint_reference']['risk_adjusted_score']:.3f}  "
            f"purity={neural['cluster_purity']:.3f}  "
            f"pred_mae={neural['mean_prediction_error']:.3f}"
        )


if __name__ == "__main__":
    main()
