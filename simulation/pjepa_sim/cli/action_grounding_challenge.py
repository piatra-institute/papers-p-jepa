"""Run the action-grounding challenge benchmark."""

from __future__ import annotations

import json

from pjepa_sim.benchmark.action_grounding import markdown_report, run_action_grounding_challenge
from pjepa_sim.paths import OUTPUT_DIR


def main() -> None:
    results = run_action_grounding_challenge()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "action_grounding_challenge.json"
    md_path = OUTPUT_DIR / "action_grounding_challenge.md"
    json_path.write_text(json.dumps(results, indent=2) + "\n")
    md_path.write_text(markdown_report(results) + "\n")
    print_report(results)
    print()
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


def print_report(results: dict) -> None:
    print("Action-grounding challenge")
    for name, task in results["tasks"].items():
        metrics = task["metrics"]
        primary = next(iter(metrics.items()))
        print(f"  {name:>38s}  {primary[0]}={_fmt(primary[1])}")


def _fmt(value) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return f"{value:.6g}"
    return str(value)


if __name__ == "__main__":
    main()
