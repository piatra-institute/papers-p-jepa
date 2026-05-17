"""Executable checks for the load-bearing KTH sample real-video benchmark."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from pjepa_sim.paths import OUTPUT_DIR
from pjepa_sim.real_video.kth_samples import run_kth_sample_benchmark


OUT = OUTPUT_DIR / "kth_sample_video_verification.json"


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
    try:
        results = run_kth_sample_benchmark(data_dir)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        print("Run: uv run python -m pjepa_sim.cli.kth_sample_video_benchmark --download", file=sys.stderr)
        sys.exit(2)

    learners = results["learners"]
    static = learners["static_appearance"]
    passive = learners["passive_next_frame"]
    motion = learners["temporal_motion"]
    dataset = results["dataset"]
    claims = [
        Claim(
            name="uses_real_video_files",
            passed=bool(dataset["real_video_files"]),
            observed=float(bool(dataset["real_video_files"])),
            threshold="= 1",
            detail="The benchmark must decode real AVI files rather than generated frames.",
        ),
        Claim(
            name="kth_sample_not_full_benchmark",
            passed=not bool(dataset["full_benchmark"]),
            observed=float(bool(dataset["full_benchmark"])),
            threshold="= 0",
            detail="The result must remain labelled as a sample smoke test, not full KTH.",
        ),
        Claim(
            name="static_appearance_beats_temporal_motion_on_sample_split",
            passed=static["accuracy"] > motion["accuracy"] + 0.15,
            observed=round(static["accuracy"] - motion["accuracy"], 6),
            threshold="> 0.15 accuracy margin",
            detail="The KTH sample split is appearance dominated and therefore does not support a motion/P-JEPA advantage claim.",
        ),
        Claim(
            name="passive_next_frame_beats_temporal_motion_on_sample_split",
            passed=passive["accuracy"] > motion["accuracy"] + 0.05,
            observed=round(passive["accuracy"] - motion["accuracy"], 6),
            threshold="> 0.05 accuracy margin",
            detail="The passive descriptor also beats temporal motion on this sample, reinforcing that this is not P-JEPA evidence.",
        ),
        Claim(
            name="temporal_motion_above_chance",
            passed=motion["accuracy"] > 0.45,
            observed=round(motion["accuracy"], 6),
            threshold="> 0.45 accuracy",
            detail="Six-class chance is about 0.167; even the weaker motion descriptor should be materially above chance.",
        ),
        Claim(
            name="real_video_result_is_not_p_jepa_evidence",
            passed=static["accuracy"] > motion["accuracy"],
            observed=round(static["accuracy"] - motion["accuracy"], 6),
            threshold="> 0",
            detail="This load-bearing real-video result should be treated as a diagnostic baseline, not as evidence that P-JEPA beats video methods.",
        ),
    ]
    report = {
        "passed": all(claim.passed for claim in claims),
        "num_claims": len(claims),
        "num_passed": sum(1 for claim in claims if claim.passed),
        "claims": [claim.as_dict() for claim in claims],
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


if __name__ == "__main__":
    main()
