"""Orchestrator for the P-JEPA hidden-regime simulation.

Run with:

    cd simulation
    uv run python -m pjepa_sim.cli.run_all

Writes ``output/results.json`` and figures under ``output/figures``. Numeric
claims in the paper should trace to keys in the JSON file.
"""

from __future__ import annotations

import json

from pjepa_sim.core.agents import agent_results, regime_prediction_table, representative_trace
from pjepa_sim.core.dishworld import DIRECT_ACTIONS, PRIOR, PROBES, REGIMES, obstruction
from pjepa_sim.core.figures import plot_agent_bars, plot_obstruction, plot_transfer
from pjepa_sim.paths import OUTPUT_DIR


OUT = OUTPUT_DIR


def main() -> None:
    (OUT / "figures").mkdir(parents=True, exist_ok=True)

    agent_list = agent_results()
    results = {
        "experiment": {
            "name": "hidden_regime_manipulation",
            "regimes": list(REGIMES),
            "direct_actions": list(DIRECT_ACTIONS),
            "probes": list(PROBES),
            "prior": {r: float(p) for r, p in zip(REGIMES, PRIOR)},
            "evaluation": "exact expectation over uniform hidden regimes and stochastic probe evidence",
        },
        "local_sections": regime_prediction_table(),
        "obstruction": {
            "prior": obstruction(PRIOR),
        },
        "agents": {agent.name: agent.as_dict() for agent in agent_list},
        "representative_trace": {
            "true_regime": "soapy",
            "steps": representative_trace("soapy"),
        },
    }

    plot_agent_bars(results, str(OUT / "figures" / "agent_outcomes.png"))
    plot_obstruction(results, str(OUT / "figures" / "obstruction_reduction.png"))
    plot_transfer(results, str(OUT / "figures" / "hidden_regime_transfer.png"))

    with (OUT / "results.json").open("w") as f:
        json.dump(results, f, indent=2)

    print("Agent outcomes:")
    for agent in agent_list:
        print(
            f"  {agent.name:>18s}  "
            f"success={agent.success_rate:.3f}  "
            f"unsafe={agent.unsafe_failure_rate:.3f}  "
            f"probes={agent.mean_probes:.3f}  "
            f"O_at_action={agent.mean_obstruction_at_action:.3f}"
        )
    print()
    print("Representative trace:")
    for step in results["representative_trace"]["steps"]:
        label = step["step"]
        obs = step["obstruction"]
        print(f"  {label:>12s}  obstruction={obs:.3f}")
    print()
    print(f"Wrote: {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
