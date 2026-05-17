"""Run the learned neural active-probing seed-sweep benchmark."""

from __future__ import annotations

import json

from pjepa_sim.paths import OUTPUT_DIR
from pjepa_sim.representation.neural_active import run_neural_active_seed_sweep_benchmark


OUT = OUTPUT_DIR


def main() -> None:
    results = run_neural_active_seed_sweep_benchmark()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "neural_active_seed_sweep_benchmark.json"
    table_path = OUT / "neural_active_seed_sweep_benchmark.md"
    path.write_text(json.dumps(results, indent=2) + "\n")
    table_path.write_text(markdown_table(results))
    print_table(results)
    print()
    print(f"Wrote: {path}")
    print(f"Wrote: {table_path}")


def markdown_table(results: dict) -> str:
    lines = [
        "| Seed | No Probe | Entropy | Active | Unsafe Reduction | Oracle Gap |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for seed, case in results["cases"].items():
        learners = case["learners"]
        no_probe = learners["learned_no_probe"]
        entropy = learners["learned_entropy_probe"]
        active = learners["learned_active_probe"]
        oracle = learners["oracle_regime"]
        lines.append(
            "| "
            f"{seed} | "
            f"{no_probe['risk_adjusted_score']:.3f} | "
            f"{entropy['risk_adjusted_score']:.3f} | "
            f"{active['risk_adjusted_score']:.3f} | "
            f"{no_probe['unsafe_failure_rate'] - active['unsafe_failure_rate']:.3f} | "
            f"{oracle['risk_adjusted_score'] - active['risk_adjusted_score']:.3f} |"
        )
    summary = results["summary"]
    lines.append(
        "| "
        "`mean` | "
        " | "
        f"{summary['active_minus_entropy_mean']:.3f} margin | "
        f"{summary['active_score_mean']:.3f} | "
        f"{summary['no_probe_minus_active_unsafe_mean']:.3f} | "
        f"{summary['oracle_gap_mean']:.3f} |"
    )
    return "\n".join(lines) + "\n"


def print_table(results: dict) -> None:
    print("Neural active-probe seed sweep")
    for seed, case in results["cases"].items():
        learners = case["learners"]
        no_probe = learners["learned_no_probe"]
        entropy = learners["learned_entropy_probe"]
        active = learners["learned_active_probe"]
        oracle = learners["oracle_regime"]
        print(
            f"  seed={int(seed):>2d}  "
            f"no_probe={no_probe['risk_adjusted_score']:.3f}  "
            f"entropy={entropy['risk_adjusted_score']:.3f}  "
            f"active={active['risk_adjusted_score']:.3f}  "
            f"unsafe_reduction={no_probe['unsafe_failure_rate'] - active['unsafe_failure_rate']:.3f}  "
            f"oracle_gap={oracle['risk_adjusted_score'] - active['risk_adjusted_score']:.3f}"
        )
    summary = results["summary"]
    print(
        "  summary  "
        f"active_mean={summary['active_score_mean']:.3f}  "
        f"active_std={summary['active_score_std']:.3f}  "
        f"min_score_margin={summary['active_minus_no_probe_min']:.3f}  "
        f"min_entropy_margin={summary['active_minus_entropy_min']:.3f}"
    )


if __name__ == "__main__":
    main()
