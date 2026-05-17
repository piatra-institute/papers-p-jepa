"""Prepare real-video manifests for benchmark runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pjepa_sim.real_video.manifest_benchmark import validate_manifest
from pjepa_sim.real_video.manifest_builders import (
    DEFAULT_KTH_TEST_SUBJECTS,
    DEFAULT_KTH_TRAIN_SUBJECTS,
    build_kth_manifest,
    parse_subject_list,
    write_manifest_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare real-video benchmark manifests.")
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    kth = subparsers.add_parser("kth", help="Build a KTH action-database manifest from downloaded video files.")
    kth.add_argument("--video-root", type=Path, required=True, help="Directory containing KTH video files.")
    kth.add_argument("--output", type=Path, required=True, help="CSV manifest path to write.")
    kth.add_argument(
        "--train-subjects",
        default=f"{DEFAULT_KTH_TRAIN_SUBJECTS[0]}-{DEFAULT_KTH_TRAIN_SUBJECTS[-1]}",
        help="Comma/range subject ids for train split, e.g. 01-16.",
    )
    kth.add_argument(
        "--test-subjects",
        default=f"{DEFAULT_KTH_TEST_SUBJECTS[0]}-{DEFAULT_KTH_TEST_SUBJECTS[-1]}",
        help="Comma/range subject ids for test split, e.g. 17-25.",
    )
    kth.add_argument(
        "--allow-invalid",
        action="store_true",
        help="Write the manifest even if validation fails and return success.",
    )

    args = parser.parse_args()
    if args.dataset == "kth":
        records = build_kth_manifest(
            args.video_root,
            train_subjects=parse_subject_list(args.train_subjects),
            test_subjects=parse_subject_list(args.test_subjects),
        )
        write_manifest_csv(records, args.output)
        validation = validate_manifest(
            records,
            video_root=args.video_root,
            require_group_split=True,
            require_action_metadata=True,
        )
        print(f"Wrote: {args.output}")
        print(f"records={len(records)} classes={len({record.label for record in records})}")
        for name, passed in validation["checks"].items():
            status = "PASS" if passed else "FAIL"
            print(f"{status:>4s} {name}")
        if not validation["passed"] and not args.allow_invalid:
            sys.exit(2)


if __name__ == "__main__":
    main()
