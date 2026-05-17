"""Manifest-based real-video benchmark runner.

This module is the bridge from the KTH sample smoke test to full real-video
datasets. It requires train/test records to be split at the video group level
so that temporal windows from the same subject, scene, or source video cannot
appear in both train and test.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pjepa_sim.paths import OUTPUT_DIR
from pjepa_sim.real_video.kth_samples import (
    STEP,
    Segment,
    WINDOW,
    decode_video,
    evaluate_feature,
    passive_next_frame_features,
    static_appearance_features,
    temporal_motion_features,
)


TRAIN_SPLITS = {"train", "training"}
TEST_SPLITS = {"test", "val", "valid", "validation"}
FEATURES = {
    "static_appearance": static_appearance_features,
    "passive_next_frame": passive_next_frame_features,
    "temporal_motion": temporal_motion_features,
}


@dataclass(frozen=True)
class VideoRecord:
    path: str
    label: str
    split: str
    group: str
    action: str = ""
    subject: str = ""
    scene: str = ""

    @property
    def normalised_split(self) -> str:
        split = self.split.strip().lower()
        if split in TRAIN_SPLITS:
            return "train"
        if split in TEST_SPLITS:
            return "test"
        return split


def run_manifest_video_benchmark(
    manifest_path: Path,
    *,
    video_root: Path,
    output_name: str = "manifest_video_benchmark",
    require_group_split: bool = True,
    require_action_metadata: bool = False,
    max_videos: int | None = None,
    max_segments_per_video: int | None = None,
) -> dict[str, Any]:
    records = read_manifest(manifest_path)
    if max_videos is not None:
        records = records[:max_videos]
    validation = validate_manifest(
        records,
        video_root=video_root,
        require_group_split=require_group_split,
        require_action_metadata=require_action_metadata,
    )
    results: dict[str, Any] = {
        "benchmark": output_name,
        "manifest": str(manifest_path),
        "video_root": str(video_root),
        "validation": validation,
        "dataset": {
            "num_videos": len(records),
            "num_classes": len({record.label for record in records}),
            "requires_group_disjoint_split": require_group_split,
            "requires_action_metadata": require_action_metadata,
        },
        "learners": {},
    }
    if not validation["passed"]:
        return results

    train, test = load_manifest_segments(
        records,
        video_root=video_root,
        max_segments_per_video=max_segments_per_video,
    )
    results["dataset"].update(
        {
            "num_train_segments": len(train),
            "num_test_segments": len(test),
            "train_classes": sorted({segment.label for segment in train}),
            "test_classes": sorted({segment.label for segment in test}),
        }
    )
    results["learners"] = {
        name: evaluate_feature(train, test, feature_fn)
        for name, feature_fn in FEATURES.items()
    }
    return results


def validate_manifest_file(
    manifest_path: Path,
    *,
    video_root: Path,
    require_group_split: bool = True,
    require_action_metadata: bool = False,
    max_videos: int | None = None,
    output_name: str = "manifest_video_validation",
) -> dict[str, Any]:
    records = read_manifest(manifest_path)
    if max_videos is not None:
        records = records[:max_videos]
    validation = validate_manifest(
        records,
        video_root=video_root,
        require_group_split=require_group_split,
        require_action_metadata=require_action_metadata,
    )
    return {
        "benchmark": output_name,
        "manifest": str(manifest_path),
        "video_root": str(video_root),
        "validation": validation,
        "dataset": {
            "num_videos": len(records),
            "num_classes": len({record.label for record in records}),
            "requires_group_disjoint_split": require_group_split,
            "requires_action_metadata": require_action_metadata,
        },
        "learners": {},
    }


def write_manifest_video_outputs(results: dict[str, Any], *, output_name: str) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / f"{output_name}.json"
    md_path = OUTPUT_DIR / f"{output_name}.md"
    json_path.write_text(json.dumps(results, indent=2) + "\n")
    md_path.write_text(_markdown(results))
    return json_path, md_path


def read_manifest(path: Path) -> list[VideoRecord]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        rows = data["videos"] if isinstance(data, dict) and "videos" in data else data
        if not isinstance(rows, list):
            raise ValueError("JSON manifest must be a list or an object with a 'videos' list.")
        return [_record_from_mapping(row) for row in rows]

    with path.open(newline="") as f:
        return [_record_from_mapping(row) for row in csv.DictReader(f)]


def validate_manifest(
    records: list[VideoRecord],
    *,
    video_root: Path,
    require_group_split: bool,
    require_action_metadata: bool,
) -> dict[str, Any]:
    checks = {
        "has_records": len(records) > 0,
        "all_files_exist": all((video_root / record.path).exists() for record in records),
        "uses_train_and_test": {"train", "test"}.issubset({record.normalised_split for record in records}),
        "all_splits_known": all(record.normalised_split in {"train", "test"} for record in records),
        "same_classes_in_train_and_test": _same_classes_in_train_and_test(records),
        "path_disjoint_train_test": _path_disjoint(records),
        "has_group_metadata": all(bool(record.group.strip()) for record in records),
        "group_disjoint_train_test": _group_disjoint(records),
        "has_action_metadata": all(bool(record.action.strip()) for record in records),
    }
    if not require_group_split:
        checks["has_group_metadata"] = True
        checks["group_disjoint_train_test"] = True
    if not require_action_metadata:
        checks["has_action_metadata"] = True
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "class_counts": _class_counts(records),
        "missing_files": [
            record.path for record in records if not (video_root / record.path).exists()
        ],
        "path_overlap": sorted(_train_paths(records) & _test_paths(records)),
        "group_overlap": sorted(_train_groups(records) & _test_groups(records)),
    }


def load_manifest_segments(
    records: list[VideoRecord],
    *,
    video_root: Path,
    max_segments_per_video: int | None,
) -> tuple[list[Segment], list[Segment]]:
    train: list[Segment] = []
    test: list[Segment] = []
    for record in records:
        frames = decode_video(video_root / record.path)
        segments = _segments_for_record(record, frames)
        if max_segments_per_video is not None:
            segments = segments[:max_segments_per_video]
        if record.normalised_split == "train":
            train.extend(segments)
        elif record.normalised_split == "test":
            test.extend(segments)
    return train, test


def _segments_for_record(record: VideoRecord, frames) -> list[Segment]:
    if len(frames) < WINDOW:
        return [
            Segment(
                label=record.label,
                video_name=record.path,
                segment_index=0,
                frames=frames,
            )
        ]
    segments: list[Segment] = []
    for index, start in enumerate(range(0, len(frames) - WINDOW + 1, STEP)):
        segments.append(
            Segment(
                label=record.label,
                video_name=record.path,
                segment_index=index,
                frames=frames[start : start + WINDOW],
            )
        )
    return segments


def _record_from_mapping(row: dict[str, Any]) -> VideoRecord:
    missing = [field for field in ("path", "label", "split") if not str(row.get(field, "")).strip()]
    if missing:
        raise ValueError(f"Manifest row is missing required fields: {', '.join(missing)}")
    group = str(row.get("group") or row.get("subject") or row.get("scene") or "")
    return VideoRecord(
        path=str(row["path"]),
        label=str(row["label"]),
        split=str(row["split"]),
        group=group,
        action=str(row.get("action", "")),
        subject=str(row.get("subject", "")),
        scene=str(row.get("scene", "")),
    )


def _same_classes_in_train_and_test(records: list[VideoRecord]) -> bool:
    train = {record.label for record in records if record.normalised_split == "train"}
    test = {record.label for record in records if record.normalised_split == "test"}
    return bool(train) and train == test


def _group_disjoint(records: list[VideoRecord]) -> bool:
    return not (_train_groups(records) & _test_groups(records))


def _path_disjoint(records: list[VideoRecord]) -> bool:
    return not (_train_paths(records) & _test_paths(records))


def _train_paths(records: list[VideoRecord]) -> set[str]:
    return {record.path for record in records if record.normalised_split == "train"}


def _test_paths(records: list[VideoRecord]) -> set[str]:
    return {record.path for record in records if record.normalised_split == "test"}


def _train_groups(records: list[VideoRecord]) -> set[str]:
    return {record.group for record in records if record.normalised_split == "train"}


def _test_groups(records: list[VideoRecord]) -> set[str]:
    return {record.group for record in records if record.normalised_split == "test"}


def _class_counts(records: list[VideoRecord]) -> dict[str, dict[str, int]]:
    labels = sorted({record.label for record in records})
    counts: dict[str, dict[str, int]] = {}
    for label in labels:
        counts[label] = {
            "train": sum(1 for record in records if record.label == label and record.normalised_split == "train"),
            "test": sum(1 for record in records if record.label == label and record.normalised_split == "test"),
        }
    return counts


def _markdown(results: dict[str, Any]) -> str:
    validation = results["validation"]
    lines = [
        "# Manifest Video Benchmark",
        "",
        f"Manifest: `{results['manifest']}`",
        "",
        f"Validation passed: `{validation['passed']}`",
        "",
        "| Check | Pass |",
        "|---|---:|",
    ]
    for name, passed in validation["checks"].items():
        lines.append(f"| `{name}` | {'yes' if passed else 'no'} |")
    if results["learners"]:
        lines.extend(
            [
                "",
                "| Learner | Accuracy | Train Segments | Test Segments | Mean Margin |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for name, metrics in results["learners"].items():
            lines.append(
                "| "
                f"`{name}` | "
                f"{metrics['accuracy']:.3f} | "
                f"{metrics['num_train_segments']} | "
                f"{metrics['num_test_segments']} | "
                f"{metrics['mean_nearest_margin']:.3f} |"
            )
    return "\n".join(lines) + "\n"
