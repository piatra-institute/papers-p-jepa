"""Audit metadata and claim-summary generation for local verifiers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pjepa_sim.paths import OUTPUT_DIR


@dataclass(frozen=True)
class VerifierSpec:
    label: str
    module: str
    report_path: Path
    limitation: str


LOCAL_VERIFIERS: tuple[VerifierSpec, ...] = (
    VerifierSpec(
        label="Exact hidden-regime mechanism",
        module="pjepa_sim.verification.exact_claims",
        report_path=OUTPUT_DIR / "verification.json",
        limitation="Small hand-specified hidden-regime world with exact expectations.",
    ),
    VerifierSpec(
        label="Suite-level P-JEPA stack",
        module="pjepa_sim.verification.benchmark_claims",
        report_path=OUTPUT_DIR / "benchmark_verification.json",
        limitation="Configured hidden-regime suites; cover, probes, and regime vocabulary remain hand-specified.",
    ),
    VerifierSpec(
        label="Action-grounded representation",
        module="pjepa_sim.verification.representation_claims",
        report_path=OUTPUT_DIR / "representation_verification.json",
        limitation="Uses engineered action/probe fingerprints, not perception from raw streams.",
    ),
    VerifierSpec(
        label="Neural intervention encoder",
        module="pjepa_sim.verification.neural_claims",
        report_path=OUTPUT_DIR / "neural_verification.json",
        limitation="Learns from low-dimensional structured sensors and test identities, not pixels or tactile streams.",
    ),
    VerifierSpec(
        label="Neural sample efficiency",
        module="pjepa_sim.verification.neural_sample_efficiency_claims",
        report_path=OUTPUT_DIR / "neural_sample_efficiency_verification.json",
        limitation="Sparse-evidence toy sweep, not a general data-efficiency result.",
    ),
    VerifierSpec(
        label="Neural active probing",
        module="pjepa_sim.verification.neural_active_probe_claims",
        report_path=OUTPUT_DIR / "neural_active_probe_verification.json",
        limitation="Structured sensor and probe-evidence features with exact evidence-tree evaluation.",
    ),
    VerifierSpec(
        label="Neural active-probing boundary",
        module="pjepa_sim.verification.neural_active_boundary_claims",
        report_path=OUTPUT_DIR / "neural_active_boundary_verification.json",
        limitation="Controlled negative cases for sensor aliasing, weak probes, and costly probes.",
    ),
    VerifierSpec(
        label="Neural active-probing seed sweep",
        module="pjepa_sim.verification.neural_active_seed_sweep_claims",
        report_path=OUTPUT_DIR / "neural_active_seed_sweep_verification.json",
        limitation="Small deterministic seed sweep; not a statistical confidence interval.",
    ),
    VerifierSpec(
        label="Pixel continuous control",
        module="pjepa_sim.verification.pixel_continuous_claims",
        report_path=OUTPUT_DIR / "pixel_continuous_verification.json",
        limitation="Small rendered-image 2D control benchmark; not raw robot perception or MuJoCo-scale control.",
    ),
    VerifierSpec(
        label="Video representation surrogate",
        module="pjepa_sim.verification.video_representation_claims",
        report_path=OUTPUT_DIR / "video_representation_verification.json",
        limitation="Local passive-video JEPA surrogate; not an actual V-JEPA or video-foundation-model benchmark.",
    ),
    VerifierSpec(
        label="KTH sample real video",
        module="pjepa_sim.verification.kth_sample_video_claims",
        report_path=OUTPUT_DIR / "kth_sample_video_verification.json",
        limitation="Load-bearing real-video smoke test using downloaded KTH sample AVI files; not full KTH, not V-JEPA, and currently diagnostic rather than positive for P-JEPA.",
    ),
    VerifierSpec(
        label="Manifest real-video protocol",
        module="pjepa_sim.verification.manifest_video_protocol_claims",
        report_path=OUTPUT_DIR / "manifest_video_protocol_verification.json",
        limitation="Protocol verifier for leakage-aware real-video manifests; it does not report a new dataset performance result.",
    ),
    VerifierSpec(
        label="Robot manifest protocol",
        module="pjepa_sim.verification.robot_manifest_protocol_claims",
        report_path=OUTPUT_DIR / "robot_manifest_protocol_verification.json",
        limitation="Protocol verifier for future robot-policy manifests; it does not report learned robot policy performance.",
    ),
    VerifierSpec(
        label="Formal contract interface",
        module="pjepa_sim.verification.formal_contract_claims",
        report_path=OUTPUT_DIR / "formal_contract_verification.json",
        limitation="Local finite-state contract checker; exports a Kona/Aleph-style verification interface but does not run those external systems.",
    ),
    VerifierSpec(
        label="Online cover construction",
        module="pjepa_sim.verification.online_claims",
        report_path=OUTPUT_DIR / "online_cover_verification.json",
        limitation="Incremental cover construction from engineered action-consequence fingerprints.",
    ),
    VerifierSpec(
        label="Synthetic regime scaling",
        module="pjepa_sim.verification.scaling_claims",
        report_path=OUTPUT_DIR / "scaling_verification.json",
        limitation="Synthetic regime-count sweep over engineered fingerprints, not high-dimensional neural scaling.",
    ),
    VerifierSpec(
        label="Restriction-map gluing",
        module="pjepa_sim.verification.gluing_claims",
        report_path=OUTPUT_DIR / "gluing_ablation_verification.json",
        limitation="Linear restriction maps over engineered local section vectors, not end-to-end neural sheaf learning.",
    ),
    VerifierSpec(
        label="Skill composition",
        module="pjepa_sim.verification.composition_claims",
        report_path=OUTPUT_DIR / "skill_composition_verification.json",
        limitation="Minimal precondition/postcondition skill-table benchmark, not learned options.",
    ),
    VerifierSpec(
        label="Action-grounding challenge",
        module="pjepa_sim.verification.action_grounding_challenge_claims",
        report_path=OUTPUT_DIR / "action_grounding_challenge_verification.json",
        limitation="Integrated practical-use harness over controlled local benchmarks; not a full robotics or video-foundation-model result.",
    ),
    VerifierSpec(
        label="Evidence-level guard",
        module="pjepa_sim.verification.evidence_claims",
        report_path=OUTPUT_DIR / "evidence_verification.json",
        limitation="Claim-boundary verifier; classifies evidence levels and prevents broad overclaims.",
    ),
)


def write_claims_summary(
    specs: tuple[VerifierSpec, ...] = LOCAL_VERIFIERS,
    *,
    output_dir: Path = OUTPUT_DIR,
) -> tuple[Path, Path]:
    rows = []
    for spec in specs:
        report = _load_report(spec.report_path)
        for claim in _extract_claims(report):
            rows.append(
                {
                    "benchmark": spec.label,
                    "claim": claim["name"],
                    "passed": bool(claim["passed"]),
                    "observed": claim["observed"],
                    "threshold": claim["threshold"],
                    "verifier_json": str(spec.report_path.relative_to(output_dir.parent)),
                    "verifier_module": spec.module,
                    "limitation": spec.limitation,
                    "detail": claim["detail"],
                }
            )

    report = {
        "passed": all(row["passed"] for row in rows),
        "num_claims": len(rows),
        "num_passed": sum(1 for row in rows if row["passed"]),
        "claims": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "claims_summary.json"
    md_path = output_dir / "CLAIMS_SUMMARY.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    md_path.write_text(_claims_markdown(report))
    return json_path, md_path


def _load_report(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _extract_claims(report: dict[str, Any]) -> list[dict[str, str | bool]]:
    if "claims" in report:
        return [
            {
                "name": str(claim.get("name", "unnamed_claim")),
                "passed": bool(claim.get("passed", False)),
                "observed": _format_value(claim.get("observed", "")),
                "threshold": _format_value(_claim_threshold(claim)),
                "detail": str(claim.get("detail", "")),
            }
            for claim in report["claims"]
        ]
    if "checks" in report:
        return [
            {
                "name": name,
                "passed": bool(check.get("passed", False)),
                "observed": _format_value(check.get("margin", "")),
                "threshold": "margin > 0",
                "detail": "Margin-based verifier check.",
            }
            for name, check in report["checks"].items()
        ]
    return [
        {
            "name": "report_passed",
            "passed": bool(report.get("passed", False)),
            "observed": _format_value(report.get("num_passed", "")),
            "threshold": _format_value(report.get("num_claims", "")),
            "detail": "Verifier report did not expose individual claims.",
        }
    ]


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, dict):
        return ", ".join(f"{key}={_format_value(item)}" for key, item in value.items())
    if isinstance(value, list):
        return "[" + ", ".join(_format_value(item) for item in value) + "]"
    return str(value)


def _claim_threshold(claim: dict[str, Any]) -> Any:
    if "threshold" in claim:
        return claim["threshold"]
    if isinstance(claim.get("observed"), dict):
        return "all listed margins > 0"
    return ""


def _claims_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Claims Summary",
        "",
        "Generated by `uv run python -m pjepa_sim.cli.verify_all`.",
        "",
        f"Passed: {report['num_passed']} / {report['num_claims']}",
        "",
        "| Benchmark | Claim | Pass | Observed | Threshold | Verifier JSON | Limitation |",
        "|---|---|---:|---|---|---|---|",
    ]
    for row in report["claims"]:
        lines.append(
            "| "
            f"{_md(row['benchmark'])} | "
            f"`{_md(row['claim'])}` | "
            f"{'yes' if row['passed'] else 'no'} | "
            f"{_md(row['observed'])} | "
            f"{_md(row['threshold'])} | "
            f"`{_md(row['verifier_json'])}` | "
            f"{_md(row['limitation'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def _md(value: Any) -> str:
    text = str(value).replace("\n", " ")
    return text.replace("|", "\\|")
