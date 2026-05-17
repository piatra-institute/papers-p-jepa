"""Executable checks for learned active-probing boundary conditions."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

from pjepa_sim.paths import OUTPUT_DIR
from pjepa_sim.representation.neural_active import run_neural_active_boundary_benchmark


OUT = OUTPUT_DIR / "neural_active_boundary_verification.json"


@dataclass(frozen=True)
class Claim:
    name: str
    passed: bool
    observed: float
    threshold: str
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
    results = run_neural_active_boundary_benchmark()
    summary = results["summary"]
    cases = results["cases"]
    ambiguous = cases["ambiguous_informative"]["summary"]
    distinct = cases["distinct_informative"]["summary"]
    weak = cases["ambiguous_weak_probe"]["summary"]
    costly = cases["ambiguous_costly_probe"]["summary"]
    claims = [
        Claim(
            name="ambiguous_informative_active_probe_has_large_margin",
            passed=summary["ambiguous_margin"] > 0.17,
            observed=round(summary["ambiguous_margin"], 6),
            threshold="> 0.17 score margin",
            detail="When sensors alias regimes and probes are informative, active probing should clearly beat no probing.",
        ),
        Claim(
            name="active_margin_is_larger_under_aliasing_than_distinct_sensors",
            passed=summary["ambiguous_margin"] > summary["distinct_margin"] + 0.10,
            observed=round(summary["ambiguous_margin"] - summary["distinct_margin"], 6),
            threshold="> 0.10 margin difference",
            detail="The active-probing advantage should shrink when the initial sensors already identify the regime.",
        ),
        Claim(
            name="weak_probes_reduce_active_margin",
            passed=summary["ambiguous_margin"] > summary["weak_probe_margin"] + 0.08,
            observed=round(summary["ambiguous_margin"] - summary["weak_probe_margin"], 6),
            threshold="> 0.08 margin difference",
            detail="The active-probing advantage should shrink when probe evidence is weak.",
        ),
        Claim(
            name="costly_probe_policy_uses_less_than_full_budget",
            passed=summary["costly_active_mean_probes"] < 1.5,
            observed=round(summary["costly_active_mean_probes"], 6),
            threshold="< 1.5 probes",
            detail="Value-aware probing should not blindly spend the full two-probe budget when probes are costly.",
        ),
        Claim(
            name="ambiguous_active_beats_entropy",
            passed=summary["ambiguous_active_minus_entropy"] > 0.005,
            observed=round(summary["ambiguous_active_minus_entropy"], 6),
            threshold="> 0.005 score margin",
            detail="Value-aware probing should beat generic entropy probing in the aliased informative setting.",
        ),
        Claim(
            name="distinct_sensors_need_less_probe_repair",
            passed=distinct["active_mean_probes"] < ambiguous["active_mean_probes"],
            observed=round(ambiguous["active_mean_probes"] - distinct["active_mean_probes"], 6),
            threshold="ambiguous probes - distinct probes > 0",
            detail="When the initial observation separates regimes, the learned policy should use fewer repair probes.",
        ),
        Claim(
            name="hidden_labels_not_used_as_features",
            passed=not bool(results["config"]["hidden_labels_used_as_features"]),
            observed=float(bool(results["config"]["hidden_labels_used_as_features"])),
            threshold="= 0",
            detail="Hidden regime labels may be used only for diagnostics and evaluation.",
        ),
    ]
    report = {
        "passed": all(claim.passed for claim in claims),
        "num_claims": len(claims),
        "num_passed": sum(1 for claim in claims if claim.passed),
        "claims": [claim.as_dict() for claim in claims],
        "case_summaries": {
            "ambiguous_informative": ambiguous,
            "distinct_informative": distinct,
            "ambiguous_weak_probe": weak,
            "ambiguous_costly_probe": costly,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")

    for claim in claims:
        status = "PASS" if claim.passed else "FAIL"
        print(f"{status}  {claim.name}")
        print(f"      observed: {claim.observed}")
        print(f"      required: {claim.threshold}")
    print()
    print(f"Wrote: {OUT}")

    if not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
