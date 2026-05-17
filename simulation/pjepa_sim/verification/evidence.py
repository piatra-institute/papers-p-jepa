"""Evidence-level registry for P-JEPA claims.

This file is intentionally conservative. It separates demonstrated local
mechanisms from diagnostic negatives and protocol-only infrastructure so that
passing verifiers cannot silently become broader scientific claims.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pjepa_sim.paths import OUTPUT_DIR


@dataclass(frozen=True)
class EvidenceSpec:
    benchmark: str
    level: str
    supports: str
    does_not_support: tuple[str, ...]
    is_performance_evidence: bool

    def as_dict(self) -> dict:
        return {
            "benchmark": self.benchmark,
            "level": self.level,
            "supports": self.supports,
            "does_not_support": list(self.does_not_support),
            "is_performance_evidence": self.is_performance_evidence,
        }


FORBIDDEN_GLOBAL_CLAIMS = (
    "scalable_jepa_replacement",
    "video_foundation_model",
    "learned_robot_policy",
    "real_robot_competence",
    "end_to_end_neural_sheaf_learning",
    "unique_cohomology_advantage",
)


EVIDENCE_SPECS: tuple[EvidenceSpec, ...] = (
    EvidenceSpec(
        benchmark="Exact hidden-regime mechanism",
        level="controlled_mechanism",
        supports="Local obstruction can drive safe probing in a small exact hidden-regime world.",
        does_not_support=("robot_policy_learning", "perception_learning", "scaling"),
        is_performance_evidence=True,
    ),
    EvidenceSpec(
        benchmark="Suite-level P-JEPA stack",
        level="controlled_mechanism",
        supports="Viability-aware active probing improves risk-adjusted score across configured hidden-regime suites.",
        does_not_support=("learned_cover_discovery", "robot_policy_learning", "foundation_model_scaling"),
        is_performance_evidence=True,
    ),
    EvidenceSpec(
        benchmark="Action-grounded representation",
        level="engineered_representation",
        supports="Action-consequence fingerprints beat appearance grouping under visual cue shift.",
        does_not_support=("raw_perception_learning", "robot_policy_learning"),
        is_performance_evidence=True,
    ),
    EvidenceSpec(
        benchmark="Neural intervention encoder",
        level="learned_structured_sensor",
        supports="A small MLP can learn predicted-test vectors from structured sensor/intervention records.",
        does_not_support=("pixel_learning", "tactile_stream_learning", "robot_policy_learning"),
        is_performance_evidence=True,
    ),
    EvidenceSpec(
        benchmark="Neural sample efficiency",
        level="learned_structured_sensor",
        supports="The structured-sensor predicted-test vector remains useful under sparse sampled intervention evidence.",
        does_not_support=("general_data_efficiency", "large_scale_scaling"),
        is_performance_evidence=True,
    ),
    EvidenceSpec(
        benchmark="Neural active probing",
        level="learned_structured_sensor",
        supports="Learned value-aware probing repairs aliased structured observations before action.",
        does_not_support=("open_world_experimentation", "robot_policy_learning"),
        is_performance_evidence=True,
    ),
    EvidenceSpec(
        benchmark="Neural active-probing boundary",
        level="boundary_condition",
        supports="The active-probing gain depends on sensor aliasing, probe informativeness, and probe cost.",
        does_not_support=("unconditional_active_probe_superiority",),
        is_performance_evidence=True,
    ),
    EvidenceSpec(
        benchmark="Neural active-probing seed sweep",
        level="robustness_check",
        supports="The no-probe and unsafe-failure margins persist across tested deterministic seeds.",
        does_not_support=("statistical_confidence_interval", "uniform_entropy_dominance"),
        is_performance_evidence=True,
    ),
    EvidenceSpec(
        benchmark="Pixel continuous control",
        level="perception_stress_test",
        supports="The mechanism survives a small rendered-pixel and continuous-control stress test with modest gain.",
        does_not_support=("real_robot_vision", "mujoco_scale_control", "closing_oracle_gap"),
        is_performance_evidence=True,
    ),
    EvidenceSpec(
        benchmark="Video representation surrogate",
        level="local_surrogate",
        supports="A local passive-video surrogate can fail action-regime recovery under visual shift.",
        does_not_support=("actual_v_jepa_comparison", "video_foundation_model", "real_video_advantage"),
        is_performance_evidence=True,
    ),
    EvidenceSpec(
        benchmark="KTH sample real video",
        level="diagnostic_negative",
        supports="The audit can process real video, and this sample split is appearance dominated.",
        does_not_support=("p_jepa_video_advantage", "full_kth_benchmark", "v_jepa_comparison"),
        is_performance_evidence=True,
    ),
    EvidenceSpec(
        benchmark="Manifest real-video protocol",
        level="protocol_only",
        supports="Future full-video manifests must pass leakage, class-coverage, group, and action-metadata checks.",
        does_not_support=("dataset_performance", "p_jepa_video_advantage"),
        is_performance_evidence=False,
    ),
    EvidenceSpec(
        benchmark="Robot manifest protocol",
        level="protocol_only",
        supports="Future robot-policy manifests must expose observations, actions, tasks, groups, success, and unsafe metrics.",
        does_not_support=("learned_robot_policy", "robot_competence", "dataset_performance"),
        is_performance_evidence=False,
    ),
    EvidenceSpec(
        benchmark="Formal contract interface",
        level="finite_contract_interface",
        supports="Finite local contracts can be exported and checked locally.",
        does_not_support=("kona_result", "aleph_result", "external_theorem_proof"),
        is_performance_evidence=False,
    ),
    EvidenceSpec(
        benchmark="Online cover construction",
        level="engineered_representation",
        supports="An action-consequence cover can be built incrementally from engineered fingerprints.",
        does_not_support=("raw_online_robot_logs", "learned_sensory_cover"),
        is_performance_evidence=True,
    ),
    EvidenceSpec(
        benchmark="Synthetic regime scaling",
        level="synthetic_scaling",
        supports="The engineered action-consequence mechanism remains coherent as synthetic regime count increases.",
        does_not_support=("high_dimensional_neural_scaling", "foundation_model_scaling"),
        is_performance_evidence=True,
    ),
    EvidenceSpec(
        benchmark="Restriction-map gluing",
        level="controlled_gluing_ablation",
        supports="Learned linear restriction maps help when local action sections use incompatible coordinate frames.",
        does_not_support=("end_to_end_neural_sheaf_learning", "unique_cohomology_advantage"),
        is_performance_evidence=True,
    ),
    EvidenceSpec(
        benchmark="Skill composition",
        level="controlled_composition",
        supports="Action-grounded representations can select intended two-step precondition/postcondition chains in a minimal table benchmark.",
        does_not_support=("learned_options", "long_horizon_robot_planning"),
        is_performance_evidence=True,
    ),
    EvidenceSpec(
        benchmark="Action-grounding challenge",
        level="integrated_practical_use_harness",
        supports="The controlled local suite jointly tests passive-representation failure, predicted-test learning, safe probe repair, gluing, and skill composition.",
        does_not_support=("robot_policy_learning", "real_video_advantage", "foundation_model_scaling"),
        is_performance_evidence=True,
    ),
)


def evidence_by_benchmark() -> dict[str, EvidenceSpec]:
    return {spec.benchmark: spec for spec in EVIDENCE_SPECS}


def write_evidence_matrix(
    *,
    output_dir: Path = OUTPUT_DIR,
    specs: tuple[EvidenceSpec, ...] = EVIDENCE_SPECS,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "forbidden_global_claims": list(FORBIDDEN_GLOBAL_CLAIMS),
        "evidence": [spec.as_dict() for spec in specs],
    }
    json_path = output_dir / "evidence_matrix.json"
    md_path = output_dir / "EVIDENCE_MATRIX.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    md_path.write_text(_markdown(report))
    return json_path, md_path


def _markdown(report: dict) -> str:
    lines = [
        "# Evidence Matrix",
        "",
        "Generated by `uv run python -m pjepa_sim.verification.evidence_claims`.",
        "",
        "Forbidden global claims:",
        "",
    ]
    for claim in report["forbidden_global_claims"]:
        lines.append(f"- `{claim}`")
    lines.extend(
        [
            "",
            "| Benchmark | Level | Performance Evidence | Supports | Does Not Support |",
            "|---|---|---:|---|---|",
        ]
    )
    for item in report["evidence"]:
        lines.append(
            "| "
            f"{_md(item['benchmark'])} | "
            f"`{_md(item['level'])}` | "
            f"{'yes' if item['is_performance_evidence'] else 'no'} | "
            f"{_md(item['supports'])} | "
            f"{_md(', '.join(item['does_not_support']))} |"
        )
    return "\n".join(lines) + "\n"


def _md(value: str) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")
