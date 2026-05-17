"""Executable checks for the full real-video manifest protocol."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from pjepa_sim.paths import OUTPUT_DIR
from pjepa_sim.real_video.kth_samples import KTH_SAMPLE_VIDEOS
from pjepa_sim.real_video.manifest_builders import build_kth_manifest
from pjepa_sim.real_video.manifest_benchmark import VideoRecord, validate_manifest


OUT = OUTPUT_DIR / "manifest_video_protocol_verification.json"


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
    data_dir = Path("data/kth_samples")
    incomplete_split = validate_manifest(
        _kth_one_video_per_class_records(),
        video_root=data_dir,
        require_group_split=True,
        require_action_metadata=True,
    )
    duplicate_split = validate_manifest(
        _kth_duplicate_train_test_records(),
        video_root=data_dir,
        require_group_split=True,
        require_action_metadata=True,
    )
    duplicate_without_action = validate_manifest(
        _kth_duplicate_train_test_records(),
        video_root=data_dir,
        require_group_split=True,
        require_action_metadata=False,
    )
    kth_builder_records = build_kth_manifest(data_dir)
    kth_builder_validation = validate_manifest(
        kth_builder_records,
        video_root=data_dir,
        require_group_split=True,
        require_action_metadata=True,
    )
    missing_group = validate_manifest(
        _records_without_group_metadata(),
        video_root=data_dir,
        require_group_split=True,
        require_action_metadata=False,
    )
    claims = [
        Claim(
            name="manifest_protocol_requires_real_files",
            passed=bool(duplicate_split["checks"]["all_files_exist"]),
            observed=float(bool(duplicate_split["checks"]["all_files_exist"])),
            threshold="= 1",
            detail="The protocol checks that every manifest row points to an existing video file.",
        ),
        Claim(
            name="manifest_protocol_rejects_class_incomplete_split",
            passed=not bool(incomplete_split["checks"]["same_classes_in_train_and_test"]),
            observed=float(bool(incomplete_split["checks"]["same_classes_in_train_and_test"])),
            threshold="= 0",
            detail="A full video benchmark must not evaluate on classes absent from the train split or vice versa.",
        ),
        Claim(
            name="manifest_protocol_rejects_same_video_train_test_leakage",
            passed=not bool(duplicate_split["checks"]["path_disjoint_train_test"]),
            observed=float(bool(duplicate_split["checks"]["path_disjoint_train_test"])),
            threshold="= 0",
            detail="The protocol rejects manifests that put the same real video file in both train and test.",
        ),
        Claim(
            name="manifest_protocol_rejects_missing_action_metadata_when_required",
            passed=not bool(duplicate_split["checks"]["has_action_metadata"]),
            observed=float(bool(duplicate_split["checks"]["has_action_metadata"])),
            threshold="= 0",
            detail="Action/intervention metadata is mandatory when the benchmark is used for P-JEPA action-grounding claims.",
        ),
        Claim(
            name="manifest_protocol_accepts_non_action_video_when_not_claiming_p_jepa",
            passed=bool(duplicate_without_action["checks"]["has_action_metadata"]),
            observed=float(bool(duplicate_without_action["checks"]["has_action_metadata"])),
            threshold="= 1",
            detail="A plain action-recognition benchmark can run without intervention metadata, but cannot support P-JEPA action-grounding claims.",
        ),
        Claim(
            name="manifest_protocol_checks_group_disjoint_split",
            passed=bool(duplicate_split["checks"]["group_disjoint_train_test"]),
            observed=float(bool(duplicate_split["checks"]["group_disjoint_train_test"])),
            threshold="= 1",
            detail="The protocol separately checks group ids, so users can enforce subject/scene/video-family disjointness.",
        ),
        Claim(
            name="manifest_protocol_rejects_missing_group_metadata",
            passed=not bool(missing_group["checks"]["has_group_metadata"]),
            observed=float(bool(missing_group["checks"]["has_group_metadata"])),
            threshold="= 0",
            detail="Group-disjoint validation requires explicit group, subject, or scene metadata.",
        ),
        Claim(
            name="kth_manifest_builder_parses_sample_files",
            passed=len(kth_builder_records) == len(KTH_SAMPLE_VIDEOS),
            observed=float(len(kth_builder_records)),
            threshold=f"= {len(KTH_SAMPLE_VIDEOS)}",
            detail="The KTH manifest builder parses official KTH-style filenames into manifest records.",
        ),
        Claim(
            name="kth_manifest_builder_marks_action_metadata",
            passed=all(record.action == record.label for record in kth_builder_records),
            observed=float(all(record.action == record.label for record in kth_builder_records)),
            threshold="= 1",
            detail="KTH labels are copied into the action column so the manifest can satisfy action-metadata checks.",
        ),
        Claim(
            name="kth_sample_manifest_is_not_a_full_split",
            passed=not bool(kth_builder_validation["passed"]),
            observed=float(bool(kth_builder_validation["passed"])),
            threshold="= 0",
            detail="The six-file KTH sample set should not pass as a full subject-disjoint train/test benchmark.",
        ),
    ]
    report = {
        "passed": all(claim.passed for claim in claims),
        "num_claims": len(claims),
        "num_passed": sum(1 for claim in claims if claim.passed),
        "claims": [claim.as_dict() for claim in claims],
        "validations": {
            "incomplete_split": incomplete_split,
            "duplicate_split": duplicate_split,
            "duplicate_without_action": duplicate_without_action,
            "missing_group": missing_group,
            "kth_builder_validation": kth_builder_validation,
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


def _kth_one_video_per_class_records() -> list[VideoRecord]:
    records: list[VideoRecord] = []
    for index, (label, filename) in enumerate(KTH_SAMPLE_VIDEOS.items()):
        records.append(
            VideoRecord(
                path=filename,
                label=label,
                split="train" if index % 2 == 0 else "test",
                group=filename,
            )
        )
    return records


def _kth_duplicate_train_test_records() -> list[VideoRecord]:
    records: list[VideoRecord] = []
    for label, filename in KTH_SAMPLE_VIDEOS.items():
        records.append(
            VideoRecord(
                path=filename,
                label=label,
                split="train",
                group=f"{label}_train",
            )
        )
        records.append(
            VideoRecord(
                path=filename,
                label=label,
                split="test",
                group=f"{label}_test",
            )
        )
    return records


def _records_without_group_metadata() -> list[VideoRecord]:
    filenames = list(KTH_SAMPLE_VIDEOS.values())
    return [
        VideoRecord(
            path=filenames[0],
            label="walking",
            split="train",
            group="",
        ),
        VideoRecord(
            path=filenames[1],
            label="walking",
            split="test",
            group="",
        ),
    ]


if __name__ == "__main__":
    main()
