"""Manifest protocol for future robot-policy benchmarks.

The current paper does not claim end-to-end robot policy learning. This module
defines the minimum dataset contract a future result must satisfy before it can
be described as robot-policy evidence rather than a toy or video-only result.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pjepa_sim.real_video.manifest_benchmark import TEST_SPLITS, TRAIN_SPLITS


TRUTHY = {"1", "true", "yes", "y", "success", "safe"}
FALSY = {"0", "false", "no", "n", "failure", "unsafe"}


@dataclass(frozen=True)
class RobotEpisodeRecord:
    episode_id: str
    task: str
    split: str
    group: str
    observation_path: str
    action_path: str
    success: str = ""
    unsafe: str = ""
    robot: str = ""
    language: str = ""

    @property
    def normalised_split(self) -> str:
        split = self.split.strip().lower()
        if split in TRAIN_SPLITS:
            return "train"
        if split in TEST_SPLITS:
            return "test"
        return split


def read_robot_manifest(path: Path) -> list[RobotEpisodeRecord]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        rows = data["episodes"] if isinstance(data, dict) and "episodes" in data else data
        if not isinstance(rows, list):
            raise ValueError("JSON robot manifest must be a list or an object with an 'episodes' list.")
        return [_record_from_mapping(row) for row in rows]

    with path.open(newline="") as f:
        return [_record_from_mapping(row) for row in csv.DictReader(f)]


def validate_robot_manifest(
    records: list[RobotEpisodeRecord],
    *,
    data_root: Path,
    require_group_split: bool = True,
    require_unsafe_metric: bool = True,
    require_language: bool = False,
    require_robot_metadata: bool = False,
) -> dict[str, Any]:
    checks = {
        "has_records": len(records) > 0,
        "uses_train_and_test": {"train", "test"}.issubset({record.normalised_split for record in records}),
        "all_splits_known": all(record.normalised_split in {"train", "test"} for record in records),
        "same_tasks_in_train_and_test": _same_tasks_in_train_and_test(records),
        "has_group_metadata": all(bool(record.group.strip()) for record in records),
        "group_disjoint_train_test": _group_disjoint(records),
        "has_observation_paths": all(bool(record.observation_path.strip()) for record in records),
        "has_action_paths": all(bool(record.action_path.strip()) for record in records),
        "observation_files_exist": all((data_root / record.observation_path).exists() for record in records),
        "action_files_exist": all((data_root / record.action_path).exists() for record in records),
        "has_success_metric": all(_is_bool_like(record.success) for record in records),
        "has_unsafe_metric": all(_is_bool_like(record.unsafe) for record in records),
        "has_language_metadata": all(bool(record.language.strip()) for record in records),
        "has_robot_metadata": all(bool(record.robot.strip()) for record in records),
    }
    if not require_group_split:
        checks["has_group_metadata"] = True
        checks["group_disjoint_train_test"] = True
    if not require_unsafe_metric:
        checks["has_unsafe_metric"] = True
    if not require_language:
        checks["has_language_metadata"] = True
    if not require_robot_metadata:
        checks["has_robot_metadata"] = True
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "task_counts": _task_counts(records),
        "missing_observation_files": [
            record.observation_path
            for record in records
            if not (data_root / record.observation_path).exists()
        ],
        "missing_action_files": [
            record.action_path
            for record in records
            if not (data_root / record.action_path).exists()
        ],
        "group_overlap": sorted(_train_groups(records) & _test_groups(records)),
    }


def validate_robot_manifest_file(
    manifest_path: Path,
    *,
    data_root: Path,
    require_group_split: bool = True,
    require_unsafe_metric: bool = True,
    require_language: bool = False,
    require_robot_metadata: bool = False,
) -> dict[str, Any]:
    records = read_robot_manifest(manifest_path)
    validation = validate_robot_manifest(
        records,
        data_root=data_root,
        require_group_split=require_group_split,
        require_unsafe_metric=require_unsafe_metric,
        require_language=require_language,
        require_robot_metadata=require_robot_metadata,
    )
    return {
        "manifest": str(manifest_path),
        "data_root": str(data_root),
        "dataset": {
            "num_episodes": len(records),
            "num_tasks": len({record.task for record in records}),
            "requires_group_disjoint_split": require_group_split,
            "requires_unsafe_metric": require_unsafe_metric,
            "requires_language": require_language,
            "requires_robot_metadata": require_robot_metadata,
        },
        "validation": validation,
    }


def _record_from_mapping(row: dict[str, Any]) -> RobotEpisodeRecord:
    missing = [field for field in ("episode_id", "task", "split") if not str(row.get(field, "")).strip()]
    if missing:
        raise ValueError(f"Robot manifest row is missing required fields: {', '.join(missing)}")
    return RobotEpisodeRecord(
        episode_id=str(row["episode_id"]),
        task=str(row["task"]),
        split=str(row["split"]),
        group=str(row.get("group") or row.get("scene") or row.get("robot") or ""),
        observation_path=str(row.get("observation_path", "")),
        action_path=str(row.get("action_path", "")),
        success=str(row.get("success", "")),
        unsafe=str(row.get("unsafe", "")),
        robot=str(row.get("robot", "")),
        language=str(row.get("language", "")),
    )


def _same_tasks_in_train_and_test(records: list[RobotEpisodeRecord]) -> bool:
    train = {record.task for record in records if record.normalised_split == "train"}
    test = {record.task for record in records if record.normalised_split == "test"}
    return bool(train) and train == test


def _group_disjoint(records: list[RobotEpisodeRecord]) -> bool:
    return not (_train_groups(records) & _test_groups(records))


def _train_groups(records: list[RobotEpisodeRecord]) -> set[str]:
    return {record.group for record in records if record.normalised_split == "train"}


def _test_groups(records: list[RobotEpisodeRecord]) -> set[str]:
    return {record.group for record in records if record.normalised_split == "test"}


def _task_counts(records: list[RobotEpisodeRecord]) -> dict[str, dict[str, int]]:
    tasks = sorted({record.task for record in records})
    counts: dict[str, dict[str, int]] = {}
    for task in tasks:
        counts[task] = {
            "train": sum(1 for record in records if record.task == task and record.normalised_split == "train"),
            "test": sum(1 for record in records if record.task == task and record.normalised_split == "test"),
        }
    return counts


def _is_bool_like(value: str) -> bool:
    return value.strip().lower() in TRUTHY | FALSY
