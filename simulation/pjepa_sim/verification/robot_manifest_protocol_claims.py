"""Executable checks for the robot-policy manifest protocol."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from pjepa_sim.paths import OUTPUT_DIR
from pjepa_sim.robot.manifest_protocol import RobotEpisodeRecord, validate_robot_manifest


OUT = OUTPUT_DIR / "robot_manifest_protocol_verification.json"


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
    data_root = Path(".")
    complete = validate_robot_manifest(_complete_records(), data_root=data_root)
    missing_action = validate_robot_manifest(_missing_action_records(), data_root=data_root)
    missing_success = validate_robot_manifest(_missing_success_records(), data_root=data_root)
    missing_unsafe = validate_robot_manifest(_missing_unsafe_records(), data_root=data_root)
    missing_unsafe_allowed = validate_robot_manifest(
        _missing_unsafe_records(),
        data_root=data_root,
        require_unsafe_metric=False,
    )
    group_leak = validate_robot_manifest(_group_leak_records(), data_root=data_root)
    incomplete_task_split = validate_robot_manifest(_incomplete_task_split_records(), data_root=data_root)
    missing_file = validate_robot_manifest(_missing_file_records(), data_root=data_root)
    claims = [
        Claim(
            name="robot_manifest_accepts_complete_protocol",
            passed=bool(complete["passed"]),
            observed=float(bool(complete["passed"])),
            threshold="= 1",
            detail="A robot-policy benchmark manifest must contain observations, actions, tasks, split groups, success, and unsafe metrics.",
        ),
        Claim(
            name="robot_manifest_rejects_missing_actions",
            passed=not bool(missing_action["checks"]["has_action_paths"]),
            observed=float(bool(missing_action["checks"]["has_action_paths"])),
            threshold="= 0",
            detail="Robot policy learning claims require action trajectories or action labels.",
        ),
        Claim(
            name="robot_manifest_rejects_missing_success_metric",
            passed=not bool(missing_success["checks"]["has_success_metric"]),
            observed=float(bool(missing_success["checks"]["has_success_metric"])),
            threshold="= 0",
            detail="Robot policy claims require a task-success metric.",
        ),
        Claim(
            name="robot_manifest_rejects_missing_unsafe_metric_by_default",
            passed=not bool(missing_unsafe["checks"]["has_unsafe_metric"]),
            observed=float(bool(missing_unsafe["checks"]["has_unsafe_metric"])),
            threshold="= 0",
            detail="P-JEPA safety claims require explicit unsafe/failure annotations by default.",
        ),
        Claim(
            name="robot_manifest_can_validate_non_safety_runs_without_unsafe",
            passed=bool(missing_unsafe_allowed["checks"]["has_unsafe_metric"]),
            observed=float(bool(missing_unsafe_allowed["checks"]["has_unsafe_metric"])),
            threshold="= 1",
            detail="A non-safety robot benchmark may omit unsafe annotations only when safety claims are disabled.",
        ),
        Claim(
            name="robot_manifest_rejects_group_leakage",
            passed=not bool(group_leak["checks"]["group_disjoint_train_test"]),
            observed=float(bool(group_leak["checks"]["group_disjoint_train_test"])),
            threshold="= 0",
            detail="Train/test groups must be disjoint to avoid scene, robot, or episode-family leakage.",
        ),
        Claim(
            name="robot_manifest_rejects_task_incomplete_split",
            passed=not bool(incomplete_task_split["checks"]["same_tasks_in_train_and_test"]),
            observed=float(bool(incomplete_task_split["checks"]["same_tasks_in_train_and_test"])),
            threshold="= 0",
            detail="Train and test splits must cover the same task set before comparing policy transfer.",
        ),
        Claim(
            name="robot_manifest_rejects_missing_observation_files",
            passed=not bool(missing_file["checks"]["observation_files_exist"]),
            observed=float(bool(missing_file["checks"]["observation_files_exist"])),
            threshold="= 0",
            detail="Manifest rows must point to existing observation data.",
        ),
    ]
    report = {
        "passed": all(claim.passed for claim in claims),
        "num_claims": len(claims),
        "num_passed": sum(1 for claim in claims if claim.passed),
        "claims": [claim.as_dict() for claim in claims],
        "validations": {
            "complete": complete,
            "missing_action": missing_action,
            "missing_success": missing_success,
            "missing_unsafe": missing_unsafe,
            "group_leak": group_leak,
            "incomplete_task_split": incomplete_task_split,
            "missing_file": missing_file,
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


def _complete_records() -> list[RobotEpisodeRecord]:
    return [
        _record("train-0", "pick", "train", "scene-train-a", success="1", unsafe="0"),
        _record("train-1", "place", "train", "scene-train-b", success="1", unsafe="0"),
        _record("test-0", "pick", "test", "scene-test-a", success="1", unsafe="0"),
        _record("test-1", "place", "test", "scene-test-b", success="0", unsafe="1"),
    ]


def _missing_action_records() -> list[RobotEpisodeRecord]:
    records = _complete_records()
    return [records[0], records[1], records[2], _record("test-1", "place", "test", "scene-test-b", action_path="")]


def _missing_success_records() -> list[RobotEpisodeRecord]:
    records = _complete_records()
    return [records[0], records[1], records[2], _record("test-1", "place", "test", "scene-test-b", success="")]


def _missing_unsafe_records() -> list[RobotEpisodeRecord]:
    records = _complete_records()
    return [records[0], records[1], records[2], _record("test-1", "place", "test", "scene-test-b", unsafe="")]


def _group_leak_records() -> list[RobotEpisodeRecord]:
    return [
        _record("train-0", "pick", "train", "shared-scene"),
        _record("test-0", "pick", "test", "shared-scene"),
    ]


def _incomplete_task_split_records() -> list[RobotEpisodeRecord]:
    return [
        _record("train-0", "pick", "train", "scene-train-a"),
        _record("test-0", "place", "test", "scene-test-a"),
    ]


def _missing_file_records() -> list[RobotEpisodeRecord]:
    return [
        _record("train-0", "pick", "train", "scene-train-a", observation_path="missing-observations.npz"),
        _record("test-0", "pick", "test", "scene-test-a"),
    ]


def _record(
    episode_id: str,
    task: str,
    split: str,
    group: str,
    *,
    observation_path: str = "pyproject.toml",
    action_path: str = "uv.lock",
    success: str = "1",
    unsafe: str = "0",
) -> RobotEpisodeRecord:
    return RobotEpisodeRecord(
        episode_id=episode_id,
        task=task,
        split=split,
        group=group,
        observation_path=observation_path,
        action_path=action_path,
        success=success,
        unsafe=unsafe,
        robot="sim",
        language=f"{task} object",
    )


if __name__ == "__main__":
    main()
