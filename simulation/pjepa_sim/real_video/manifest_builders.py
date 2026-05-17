"""Dataset-specific manifest builders for real-video benchmarks."""

from __future__ import annotations

import csv
import re
from pathlib import Path

from pjepa_sim.real_video.manifest_benchmark import VideoRecord


VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".webm"}
KTH_FILENAME = re.compile(
    r"^person(?P<subject>\d+)_(?P<label>[a-z]+)_d(?P<scene>\d+)_uncomp\.(?P<ext>avi|mp4|mov|mkv|webm)$",
    re.IGNORECASE,
)
DEFAULT_KTH_TRAIN_SUBJECTS = tuple(f"{index:02d}" for index in range(1, 17))
DEFAULT_KTH_TEST_SUBJECTS = tuple(f"{index:02d}" for index in range(17, 26))


def build_kth_manifest(
    video_root: Path,
    *,
    train_subjects: set[str] | None = None,
    test_subjects: set[str] | None = None,
) -> list[VideoRecord]:
    train_subjects = train_subjects or set(DEFAULT_KTH_TRAIN_SUBJECTS)
    test_subjects = test_subjects or set(DEFAULT_KTH_TEST_SUBJECTS)
    overlap = train_subjects & test_subjects
    if overlap:
        raise ValueError(f"KTH train/test subject lists overlap: {', '.join(sorted(overlap))}")

    records: list[VideoRecord] = []
    for path in sorted(_iter_video_files(video_root)):
        match = KTH_FILENAME.match(path.name)
        if not match:
            continue
        subject = match.group("subject")
        split = ""
        if subject in train_subjects:
            split = "train"
        elif subject in test_subjects:
            split = "test"
        else:
            continue
        label = match.group("label").lower()
        scene = f"d{match.group('scene')}"
        records.append(
            VideoRecord(
                path=str(path.relative_to(video_root)),
                label=label,
                split=split,
                group=f"person{subject}",
                action=label,
                subject=f"person{subject}",
                scene=scene,
            )
        )
    return records


def write_manifest_csv(records: list[VideoRecord], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["path", "label", "split", "group", "action", "subject", "scene"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "path": record.path,
                    "label": record.label,
                    "split": record.split,
                    "group": record.group,
                    "action": record.action,
                    "subject": record.subject,
                    "scene": record.scene,
                }
            )
    return path


def parse_subject_list(value: str) -> set[str]:
    subjects: set[str] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            for index in range(int(start), int(end) + 1):
                subjects.add(f"{index:02d}")
        else:
            subjects.add(f"{int(part):02d}")
    return subjects


def _iter_video_files(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path
