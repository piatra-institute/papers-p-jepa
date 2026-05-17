"""Executable checks for evidence-level claim boundaries."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

from pjepa_sim.paths import OUTPUT_DIR
from pjepa_sim.verification.audit import LOCAL_VERIFIERS
from pjepa_sim.verification.evidence import (
    FORBIDDEN_GLOBAL_CLAIMS,
    evidence_by_benchmark,
    write_evidence_matrix,
)


OUT = OUTPUT_DIR / "evidence_verification.json"


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
    specs = evidence_by_benchmark()
    verifier_labels = {spec.label for spec in LOCAL_VERIFIERS if spec.label != "Evidence-level guard"}
    missing = sorted(verifier_labels - set(specs))
    extra = sorted(set(specs) - verifier_labels)
    claims = [
        Claim(
            name="every_local_verifier_has_evidence_level",
            passed=not missing,
            observed=float(len(missing)),
            threshold="= 0 missing evidence specs",
            detail=f"Missing evidence specs: {', '.join(missing) if missing else 'none'}.",
        ),
        Claim(
            name="no_unregistered_evidence_levels",
            passed=not extra,
            observed=float(len(extra)),
            threshold="= 0 extra evidence specs",
            detail=f"Extra evidence specs: {', '.join(extra) if extra else 'none'}.",
        ),
        Claim(
            name="protocol_checks_are_not_performance_evidence",
            passed=all(
                not spec.is_performance_evidence
                for spec in specs.values()
                if spec.level == "protocol_only"
            ),
            observed=float(
                sum(
                    int(spec.is_performance_evidence)
                    for spec in specs.values()
                    if spec.level == "protocol_only"
                )
            ),
            threshold="= 0 protocol performance claims",
            detail="Manifest protocol checks must remain infrastructure, not benchmark performance results.",
        ),
        Claim(
            name="kth_sample_is_diagnostic_negative",
            passed=specs["KTH sample real video"].level == "diagnostic_negative"
            and "p_jepa_video_advantage" in specs["KTH sample real video"].does_not_support,
            observed=float(specs["KTH sample real video"].level == "diagnostic_negative"),
            threshold="= 1 diagnostic negative",
            detail="The KTH sample result must not be treated as positive P-JEPA video evidence.",
        ),
        Claim(
            name="video_surrogate_is_not_v_jepa_evidence",
            passed="actual_v_jepa_comparison" in specs["Video representation surrogate"].does_not_support,
            observed=float("actual_v_jepa_comparison" in specs["Video representation surrogate"].does_not_support),
            threshold="= 1",
            detail="The local video surrogate must not be reported as a V-JEPA comparison.",
        ),
        Claim(
            name="robot_protocol_is_not_robot_policy_result",
            passed=not specs["Robot manifest protocol"].is_performance_evidence
            and "learned_robot_policy" in specs["Robot manifest protocol"].does_not_support,
            observed=float(not specs["Robot manifest protocol"].is_performance_evidence),
            threshold="= 1 protocol only",
            detail="Robot manifest validation is a future-evidence guard, not learned robot policy performance.",
        ),
        Claim(
            name="forbidden_global_claims_are_listed",
            passed=len(FORBIDDEN_GLOBAL_CLAIMS) >= 6,
            observed=float(len(FORBIDDEN_GLOBAL_CLAIMS)),
            threshold=">= 6 forbidden global claims",
            detail="The evidence matrix explicitly lists broad claims not established by the current repository.",
        ),
    ]
    matrix_json, matrix_md = write_evidence_matrix()
    report = {
        "passed": all(claim.passed for claim in claims),
        "num_claims": len(claims),
        "num_passed": sum(1 for claim in claims if claim.passed),
        "claims": [claim.as_dict() for claim in claims],
        "evidence_matrix_json": str(matrix_json),
        "evidence_matrix_md": str(matrix_md),
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
    print(f"Wrote: {matrix_json}")
    print(f"Wrote: {matrix_md}")
    if not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
