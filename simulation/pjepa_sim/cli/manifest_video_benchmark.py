"""Run a manifest-based real-video benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pjepa_sim.real_video.manifest_benchmark import (
    run_manifest_video_benchmark,
    validate_manifest_file,
    write_manifest_video_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a group-disjoint real-video manifest benchmark.")
    parser.add_argument("--manifest", type=Path, required=True, help="CSV or JSON manifest with path,label,split columns.")
    parser.add_argument("--video-root", type=Path, default=Path("."), help="Directory relative to which manifest video paths are resolved.")
    parser.add_argument("--output-name", default="manifest_video_benchmark", help="Output filename stem under simulation/output/.")
    parser.add_argument("--allow-group-leakage", action="store_true", help="Do not require train/test group disjointness.")
    parser.add_argument("--require-action-metadata", action="store_true", help="Require an action column for every record.")
    parser.add_argument("--max-videos", type=int, default=None, help="Optional limit for fast protocol checks.")
    parser.add_argument("--max-segments-per-video", type=int, default=None, help="Optional segment cap per video.")
    parser.add_argument("--validate-only", action="store_true", help="Validate the manifest without decoding videos.")
    parser.add_argument("--allow-invalid", action="store_true", help="Write validation output without returning a failing exit code.")
    args = parser.parse_args()

    if args.validate_only:
        results = validate_manifest_file(
            args.manifest,
            video_root=args.video_root,
            output_name=args.output_name,
            require_group_split=not args.allow_group_leakage,
            require_action_metadata=args.require_action_metadata,
            max_videos=args.max_videos,
        )
    else:
        results = run_manifest_video_benchmark(
            args.manifest,
            video_root=args.video_root,
            output_name=args.output_name,
            require_group_split=not args.allow_group_leakage,
            require_action_metadata=args.require_action_metadata,
            max_videos=args.max_videos,
            max_segments_per_video=args.max_segments_per_video,
        )
    json_path, md_path = write_manifest_video_outputs(results, output_name=args.output_name)
    print("Manifest video benchmark")
    print(f"  validation_passed={results['validation']['passed']}")
    for name, passed in results["validation"]["checks"].items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status:>4s} {name}")
    if results["learners"]:
        for name, metrics in results["learners"].items():
            print(f"  {name:>20s} accuracy={metrics['accuracy']:.3f}")
    print()
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")
    if not results["validation"]["passed"] and not args.allow_invalid:
        sys.exit(2)


if __name__ == "__main__":
    main()
