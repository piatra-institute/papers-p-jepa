"""Run the load-bearing real-video KTH sample benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from pjepa_sim.real_video.kth_samples import run_kth_sample_benchmark, write_kth_sample_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the load-bearing KTH sample real-video benchmark.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/kth_samples"),
        help="Directory containing the official KTH sample AVI files.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the six official KTH sample AVI files before running.",
    )
    args = parser.parse_args()

    results = run_kth_sample_benchmark(args.data_dir, allow_download=args.download)
    json_path, md_path = write_kth_sample_outputs(results)
    print("KTH sample real-video benchmark")
    for name, metrics in results["learners"].items():
        print(
            f"  {name:>20s}  "
            f"accuracy={metrics['accuracy']:.3f}  "
            f"train={metrics['num_train_segments']}  "
            f"test={metrics['num_test_segments']}  "
            f"margin={metrics['mean_nearest_margin']:.3f}"
        )
    print()
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
