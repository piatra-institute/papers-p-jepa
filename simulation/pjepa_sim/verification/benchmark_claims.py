"""Executable checks for the benchmark-level P-JEPA stack claims."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

from pjepa_sim.benchmark.suites import available_suites, evaluate_suite, load_spec
from pjepa_sim.paths import OUTPUT_DIR


OUT = OUTPUT_DIR / "benchmark_verification.json"


@dataclass(frozen=True)
class Claim:
    name: str
    passed: bool
    observed: dict[str, float]
    detail: str

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "observed": self.observed,
            "detail": self.detail,
        }


def main() -> None:
    suites = {name: evaluate_suite(load_spec(name)) for name in available_suites()}
    claims = [
        _claim_stack_beats_prior(suites),
        _claim_stack_beats_sheaf(suites),
        _claim_stack_beats_entropy(suites),
        _claim_stack_reduces_unsafe_vs_prior(suites),
        _claim_stack_uses_fewer_probes_than_sheaf(suites),
        _claim_costly_probe_boundary_fixed(suites),
        _claim_miscalibrated_sections_handled(suites),
    ]

    summary = {
        "passed": all(claim.passed for claim in claims),
        "num_claims": len(claims),
        "num_passed": sum(1 for claim in claims if claim.passed),
        "claims": [claim.as_dict() for claim in claims],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        json.dump(summary, f, indent=2)

    for claim in claims:
        status = "PASS" if claim.passed else "FAIL"
        print(f"{status}  {claim.name}")
        for suite, value in claim.observed.items():
            print(f"      {suite}: {value:.3f}")
    print()
    print(f"Wrote: {OUT}")

    if not summary["passed"]:
        sys.exit(1)


def _claim_stack_beats_prior(suites: dict) -> Claim:
    margins = {
        name: suite["agents"]["p_jepa_stack"]["risk_adjusted_score"]
        - suite["agents"]["model_based_prior"]["risk_adjusted_score"]
        for name, suite in suites.items()
    }
    return Claim(
        name="p_jepa_stack_beats_prior_on_risk_adjusted_score",
        passed=all(value > 0.0 for value in margins.values()),
        observed=margins,
        detail="The full stack must beat the prior predictive baseline in every suite.",
    )


def _claim_stack_beats_sheaf(suites: dict) -> Claim:
    margins = {
        name: suite["agents"]["p_jepa_stack"]["risk_adjusted_score"]
        - suite["agents"]["sheaf_probe"]["risk_adjusted_score"]
        for name, suite in suites.items()
    }
    return Claim(
        name="p_jepa_stack_beats_sheaf_only_on_risk_adjusted_score",
        passed=all(value > 0.0 for value in margins.values()),
        observed=margins,
        detail="Adding viability-aware value of information should improve over pure obstruction reduction.",
    )


def _claim_stack_beats_entropy(suites: dict) -> Claim:
    margins = {
        name: suite["agents"]["p_jepa_stack"]["risk_adjusted_score"]
        - suite["agents"]["entropy_probe"]["risk_adjusted_score"]
        for name, suite in suites.items()
    }
    return Claim(
        name="p_jepa_stack_beats_entropy_probe_on_risk_adjusted_score",
        passed=all(value > 0.0 for value in margins.values()),
        observed=margins,
        detail="The full stack should beat generic posterior-entropy probing when action risk and probe cost matter.",
    )


def _claim_stack_reduces_unsafe_vs_prior(suites: dict) -> Claim:
    margins = {
        name: suite["agents"]["model_based_prior"]["unsafe_failure_rate"]
        - suite["agents"]["p_jepa_stack"]["unsafe_failure_rate"]
        for name, suite in suites.items()
    }
    return Claim(
        name="p_jepa_stack_reduces_unsafe_failure_vs_prior",
        passed=all(value > 0.0 for value in margins.values()),
        observed=margins,
        detail="The full stack must reduce unsafe failure, not only increase success.",
    )


def _claim_stack_uses_fewer_probes_than_sheaf(suites: dict) -> Claim:
    margins = {
        name: suite["agents"]["sheaf_probe"]["mean_probes"]
        - suite["agents"]["p_jepa_stack"]["mean_probes"]
        for name, suite in suites.items()
    }
    return Claim(
        name="p_jepa_stack_uses_fewer_probes_than_sheaf_only",
        passed=all(value > 0.0 for value in margins.values()),
        observed=margins,
        detail="The full stack should stop probing when the expected action value no longer justifies information gathering.",
    )


def _claim_costly_probe_boundary_fixed(suites: dict) -> Claim:
    suite = suites["costly_probe_v0"]
    margins = {
        "p_jepa_stack_minus_prior": (
            suite["agents"]["p_jepa_stack"]["risk_adjusted_score"]
            - suite["agents"]["model_based_prior"]["risk_adjusted_score"]
        ),
        "p_jepa_stack_minus_sheaf": (
            suite["agents"]["p_jepa_stack"]["risk_adjusted_score"]
            - suite["agents"]["sheaf_probe"]["risk_adjusted_score"]
        ),
    }
    return Claim(
        name="p_jepa_stack_fixes_costly_probe_boundary",
        passed=all(value > 0.0 for value in margins.values()),
        observed=margins,
        detail="The costly-probe suite should punish pure obstruction reduction but not viability-aware probing.",
    )


def _claim_miscalibrated_sections_handled(suites: dict) -> Claim:
    suite = suites["miscalibrated_sections_v0"]
    belief_model = suite["suite"]["belief_model"]
    margins = {
        "belief_action_model_differs": float(belief_model["action_model_differs_from_world"]),
        "belief_probe_model_differs": float(belief_model["probe_likelihood_differs_from_world"]),
        "p_jepa_stack_minus_prior": (
            suite["agents"]["p_jepa_stack"]["risk_adjusted_score"]
            - suite["agents"]["model_based_prior"]["risk_adjusted_score"]
        ),
        "p_jepa_stack_minus_entropy": (
            suite["agents"]["p_jepa_stack"]["risk_adjusted_score"]
            - suite["agents"]["entropy_probe"]["risk_adjusted_score"]
        ),
    }
    return Claim(
        name="p_jepa_stack_handles_miscalibrated_learned_sections",
        passed=all(value > 0.0 for value in margins.values()),
        observed=margins,
        detail="The harder suite should use a distinct learned belief model and still reward viability-aware probing.",
    )


if __name__ == "__main__":
    main()
