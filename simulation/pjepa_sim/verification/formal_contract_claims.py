"""Executable checks for the formal verification interface benchmark."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

from pjepa_sim.formal.contracts import run_formal_contract_benchmark, write_formal_contract_outputs
from pjepa_sim.paths import OUTPUT_DIR


OUT = OUTPUT_DIR / "formal_contract_verification.json"


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
    results = run_formal_contract_benchmark()
    write_formal_contract_outputs(results)
    summary = results["summary"]
    suites = len(results["suites"])
    passed_by_agent = summary["passed_by_agent"]
    counterexamples_by_agent = summary["counterexamples_by_agent"]
    p_jepa_passed = passed_by_agent["p_jepa_stack"]
    prior_passed = passed_by_agent["model_based_prior"]
    entropy_passed = passed_by_agent["entropy_probe"]
    claims = [
        Claim(
            name="p_jepa_stack_contracts_pass_all_suites",
            passed=p_jepa_passed == suites,
            observed=float(p_jepa_passed),
            threshold=f"= {suites}",
            detail="P-JEPA should satisfy the finite safety, branch-safety, score, obstruction, and probe-budget contracts on every configured suite.",
        ),
        Claim(
            name="p_jepa_stack_passes_more_contracts_than_prior",
            passed=p_jepa_passed - prior_passed >= 3,
            observed=float(p_jepa_passed - prior_passed),
            threshold=">= 3 suite contracts",
            detail="The verification interface should distinguish active representation repair from acting on the prior model.",
        ),
        Claim(
            name="p_jepa_stack_passes_more_contracts_than_entropy",
            passed=p_jepa_passed - entropy_passed >= 2,
            observed=float(p_jepa_passed - entropy_passed),
            threshold=">= 2 suite contracts",
            detail="Value-aware active probing should satisfy more finite contracts than entropy probing under the chosen safety-efficiency budget.",
        ),
        Claim(
            name="checker_returns_prior_counterexamples",
            passed=counterexamples_by_agent["model_based_prior"] > 0,
            observed=float(counterexamples_by_agent["model_based_prior"]),
            threshold="> 0 counterexamples",
            detail="The checker should return machine-readable contract violations for unsafe or low-score baselines.",
        ),
        Claim(
            name="contract_export_is_machine_readable",
            passed=summary["num_contracts"] == suites * len(results["agents"]),
            observed=float(summary["num_contracts"]),
            threshold=f"= {suites * len(results['agents'])}",
            detail="Every selected suite-agent pair should produce one finite-state contract artifact.",
        ),
        Claim(
            name="external_kona_aleph_results_are_not_claimed",
            passed=len(results["external_backends_executed"]) == 0,
            observed=float(len(results["external_backends_executed"])),
            threshold="= 0",
            detail="The benchmark is an adapter protocol and local checker; it must not report proprietary Kona or Aleph results without actually running them.",
        ),
    ]
    report = {
        "passed": all(claim.passed for claim in claims),
        "num_claims": len(claims),
        "num_passed": sum(1 for claim in claims if claim.passed),
        "claims": [claim.as_dict() for claim in claims],
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

