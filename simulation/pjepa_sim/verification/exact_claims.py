"""Executable verification checks for the P-JEPA simulation claims.

This is not a general proof that P-JEPA works in every environment. It is a
deterministic, falsifiable check that the hidden-regime simulation exhibits the
mechanism claimed in the paper.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

import numpy as np

from pjepa_sim.core.agents import (
    agent_results,
    choose_probe,
    expected_probe_reduction,
    representative_trace,
)
from pjepa_sim.core.dishworld import PRIOR, REGIMES, obstruction, prediction_matrix
from pjepa_sim.paths import OUTPUT_DIR


OUT = OUTPUT_DIR / "verification.json"


@dataclass(frozen=True)
class Claim:
    name: str
    passed: bool
    observed: float | str | list[float]
    threshold: float | str
    detail: str

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "observed": self.observed,
            "threshold": self.threshold,
            "detail": self.detail,
        }


def main() -> None:
    agents = {agent.name: agent for agent in agent_results()}
    sheaf = agents["sheaf_probe"]
    no_probe = {name: agent for name, agent in agents.items() if name != "sheaf_probe"}
    prior = agents["model_based_prior"]

    best_no_probe_success = max(agent.success_rate for agent in no_probe.values())
    best_no_probe_unsafe = min(agent.unsafe_failure_rate for agent in no_probe.values())
    prior_obstruction = obstruction(PRIOR)
    trace = representative_trace("soapy")
    trace_obstructions = [float(step["obstruction"]) for step in trace]
    first_probe = choose_probe(PRIOR, ())
    first_probe_reduction = expected_probe_reduction(PRIOR, first_probe) if first_probe else 0.0
    pairwise_distance = min_pairwise_section_distance()

    claims = [
        Claim(
            name="hidden_regimes_have_distinct_action_consequences",
            passed=pairwise_distance > 0.30,
            observed=round(pairwise_distance, 6),
            threshold="> 0.30",
            detail=(
                "The local sections differ enough that one visible class hides "
                "multiple action regimes."
            ),
        ),
        Claim(
            name="initial_obstruction_is_nonzero",
            passed=prior_obstruction > 0.20,
            observed=round(prior_obstruction, 6),
            threshold="> 0.20",
            detail="The prior has unresolved disagreement among local sections.",
        ),
        Claim(
            name="first_probe_has_positive_expected_obstruction_reduction",
            passed=first_probe is not None and first_probe_reduction > 0.0,
            observed=f"{first_probe}: {first_probe_reduction:.6f}",
            threshold="> 0",
            detail="The selected probe is justified by expected obstruction reduction.",
        ),
        Claim(
            name="sheaf_probe_improves_success_over_best_no_probe_baseline",
            passed=sheaf.success_rate > best_no_probe_success + 0.05,
            observed=round(sheaf.success_rate - best_no_probe_success, 6),
            threshold="> 0.05",
            detail="The obstruction-driven policy must beat the strongest no-probe success rate.",
        ),
        Claim(
            name="sheaf_probe_reduces_unsafe_failure",
            passed=sheaf.unsafe_failure_rate < best_no_probe_unsafe - 0.03,
            observed=round(best_no_probe_unsafe - sheaf.unsafe_failure_rate, 6),
            threshold="> 0.03 reduction",
            detail="The obstruction-driven policy must reduce unsafe failure, not only raise success.",
        ),
        Claim(
            name="sheaf_probe_reduces_obstruction_before_action",
            passed=sheaf.mean_obstruction_at_action < prior.mean_obstruction_at_action - 0.15,
            observed=round(prior.mean_obstruction_at_action - sheaf.mean_obstruction_at_action, 6),
            threshold="> 0.15 reduction",
            detail="The policy must repair representation before task action.",
        ),
        Claim(
            name="sheaf_probe_actually_uses_probes",
            passed=sheaf.mean_probes > 1.0,
            observed=round(sheaf.mean_probes, 6),
            threshold="> 1.0",
            detail="The gain must come from intervention, not from a hidden baseline difference.",
        ),
        Claim(
            name="representative_trace_obstruction_is_monotone",
            passed=is_monotone_nonincreasing(trace_obstructions),
            observed=[round(v, 6) for v in trace_obstructions],
            threshold="monotone nonincreasing",
            detail="The representative trace must show repair rather than accidental success.",
        ),
        Claim(
            name="probe_policy_changes_action_in_hidden_risk_regimes",
            passed=(
                sheaf.by_regime["cracked"]["action"] == "lift_slow"
                and sheaf.by_regime["heavy"]["action"] == "grip_hard"
            ),
            observed=(
                f"cracked->{sheaf.by_regime['cracked']['action']}, "
                f"heavy->{sheaf.by_regime['heavy']['action']}"
            ),
            threshold="cracked->lift_slow, heavy->grip_hard",
            detail="Representation repair should change the selected action where the prior is unsafe.",
        ),
    ]

    summary = {
        "passed": all(claim.passed for claim in claims),
        "num_claims": len(claims),
        "num_passed": sum(1 for claim in claims if claim.passed),
        "metrics": {
            "best_no_probe_success": best_no_probe_success,
            "best_no_probe_unsafe": best_no_probe_unsafe,
            "sheaf_success": sheaf.success_rate,
            "sheaf_unsafe": sheaf.unsafe_failure_rate,
            "prior_obstruction_at_action": prior.mean_obstruction_at_action,
            "sheaf_obstruction_at_action": sheaf.mean_obstruction_at_action,
            "sheaf_mean_probes": sheaf.mean_probes,
        },
        "claims": [claim.as_dict() for claim in claims],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        json.dump(summary, f, indent=2)

    for claim in claims:
        status = "PASS" if claim.passed else "FAIL"
        print(f"{status}  {claim.name}")
        print(f"      observed: {claim.observed}")
        print(f"      required: {claim.threshold}")

    print()
    print(f"Wrote: {OUT}")

    if not summary["passed"]:
        sys.exit(1)


def min_pairwise_section_distance() -> float:
    preds = prediction_matrix()
    distances = []
    for i in range(len(REGIMES)):
        for j in range(i + 1, len(REGIMES)):
            distances.append(float(np.linalg.norm(preds[i] - preds[j])))
    return min(distances)


def is_monotone_nonincreasing(values: list[float]) -> bool:
    return all(a >= b for a, b in zip(values, values[1:]))


if __name__ == "__main__":
    main()
