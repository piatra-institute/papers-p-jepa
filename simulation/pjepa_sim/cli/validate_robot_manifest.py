"""Validate a robot/action dataset manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pjepa_sim.paths import OUTPUT_DIR
from pjepa_sim.robot.manifest_protocol import validate_robot_manifest_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a robot/action manifest before policy-learning claims.")
    parser.add_argument("--manifest", type=Path, required=True, help="CSV or JSON manifest with robot episode records.")
    parser.add_argument("--data-root", type=Path, default=Path("."), help="Directory relative to which manifest paths are resolved.")
    parser.add_argument("--output-name", default="robot_manifest_validation", help="Output filename stem under simulation/output/.")
    parser.add_argument("--allow-group-leakage", action="store_true", help="Do not require train/test group disjointness.")
    parser.add_argument("--allow-missing-unsafe", action="store_true", help="Do not require an unsafe/failure metric.")
    parser.add_argument("--require-language", action="store_true", help="Require language metadata for every episode.")
    parser.add_argument("--require-robot-metadata", action="store_true", help="Require robot/embodiment metadata for every episode.")
    parser.add_argument("--allow-invalid", action="store_true", help="Write validation output without returning a failing exit code.")
    args = parser.parse_args()

    report = validate_robot_manifest_file(
        args.manifest,
        data_root=args.data_root,
        require_group_split=not args.allow_group_leakage,
        require_unsafe_metric=not args.allow_missing_unsafe,
        require_language=args.require_language,
        require_robot_metadata=args.require_robot_metadata,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{args.output_name}.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print("Robot manifest validation")
    print(f"  validation_passed={report['validation']['passed']}")
    for name, passed in report["validation"]["checks"].items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status:>4s} {name}")
    print()
    print(f"Wrote: {output_path}")
    if not report["validation"]["passed"] and not args.allow_invalid:
        sys.exit(2)


if __name__ == "__main__":
    main()
