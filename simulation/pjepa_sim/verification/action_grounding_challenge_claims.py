"""Executable checks for the action-grounding practical-use challenge."""

from __future__ import annotations

import json
import sys

from pjepa_sim.benchmark.action_grounding import (
    evaluate_action_grounding_claims,
    run_action_grounding_challenge,
)
from pjepa_sim.paths import OUTPUT_DIR


OUT = OUTPUT_DIR / "action_grounding_challenge_verification.json"


def main() -> None:
    results = run_action_grounding_challenge()
    claims = evaluate_action_grounding_claims(results)
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
