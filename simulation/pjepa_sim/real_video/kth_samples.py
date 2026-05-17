"""Real-video KTH sample benchmark.

This load-bearing benchmark uses the official KTH action database sample
videos, not generated frames. It is intentionally small: one public sample AVI
for each of six KTH action classes. Segments are split within each sample video,
so this is a smoke test for real video processing, not a publishable KTH
benchmark.
"""

from __future__ import annotations

import json
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pjepa_sim.paths import OUTPUT_DIR


DATASET_URL = "https://www.csc.kth.se/cvap/actions/"
KTH_SAMPLE_VIDEOS = {
    "walking": "person15_walking_d1_uncomp.avi",
    "jogging": "person15_jogging_d1_uncomp.avi",
    "running": "person15_running_d1_uncomp.avi",
    "boxing": "person15_boxing_d1_uncomp.avi",
    "handwaving": "person15_handwaving_d1_uncomp.avi",
    "handclapping": "person15_handclapping_d1_uncomp.avi",
}
FRAME_WIDTH = 64
FRAME_HEIGHT = 48
FPS = 12
WINDOW = 16
STEP = 8


@dataclass(frozen=True)
class Segment:
    label: str
    video_name: str
    segment_index: int
    frames: np.ndarray


def run_kth_sample_benchmark(
    data_dir: Path,
    *,
    allow_download: bool = False,
) -> dict[str, Any]:
    data_dir = Path(data_dir)
    if allow_download:
        download_kth_samples(data_dir)
    missing = missing_videos(data_dir)
    if missing:
        raise FileNotFoundError(
            "Missing KTH sample videos: "
            + ", ".join(missing)
            + f". Run with --download or place files under {data_dir}."
        )

    segments = load_segments(data_dir)
    train, test = split_segments(segments)
    learners = {
        "static_appearance": evaluate_feature(train, test, static_appearance_features),
        "passive_next_frame": evaluate_feature(train, test, passive_next_frame_features),
        "temporal_motion": evaluate_feature(train, test, temporal_motion_features),
    }
    return {
        "benchmark": "kth_sample_real_video",
        "description": (
            "Load-bearing real-video smoke test using official KTH action database sample AVI files. "
            "This is not the full KTH benchmark and not an intervention/P-JEPA robotics dataset."
        ),
        "dataset": {
            "name": "KTH action database sample videos",
            "source_url": DATASET_URL,
            "classes": list(KTH_SAMPLE_VIDEOS),
            "videos": KTH_SAMPLE_VIDEOS,
            "num_videos": len(KTH_SAMPLE_VIDEOS),
            "num_segments": len(segments),
            "num_train_segments": len(train),
            "num_test_segments": len(test),
            "split": "alternating temporal windows within each sample video",
            "full_benchmark": False,
            "real_video_files": True,
        },
        "learners": learners,
    }


def write_kth_sample_outputs(results: dict[str, Any]) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "kth_sample_video_benchmark.json"
    md_path = OUTPUT_DIR / "kth_sample_video_benchmark.md"
    json_path.write_text(json.dumps(results, indent=2) + "\n")
    md_path.write_text(_markdown(results))
    return json_path, md_path


def download_kth_samples(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename in KTH_SAMPLE_VIDEOS.values():
        target = data_dir / filename
        if target.exists() and target.stat().st_size > 0:
            continue
        urllib.request.urlretrieve(DATASET_URL + filename, target)


def missing_videos(data_dir: Path) -> list[str]:
    return [
        filename
        for filename in KTH_SAMPLE_VIDEOS.values()
        if not (data_dir / filename).exists()
    ]


def load_segments(data_dir: Path) -> list[Segment]:
    segments: list[Segment] = []
    for label, filename in KTH_SAMPLE_VIDEOS.items():
        frames = decode_video(data_dir / filename)
        if len(frames) < WINDOW:
            continue
        index = 0
        for start in range(0, len(frames) - WINDOW + 1, STEP):
            segments.append(
                Segment(
                    label=label,
                    video_name=filename,
                    segment_index=index,
                    frames=frames[start : start + WINDOW],
                )
            )
            index += 1
    return segments


def decode_video(path: Path) -> np.ndarray:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-vf",
        f"fps={FPS},scale={FRAME_WIDTH}:{FRAME_HEIGHT},format=gray",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]
    completed = subprocess.run(command, check=True, capture_output=True)
    frame_size = FRAME_WIDTH * FRAME_HEIGHT
    raw = completed.stdout
    n_frames = len(raw) // frame_size
    if n_frames == 0:
        raise ValueError(f"ffmpeg decoded no frames from {path}")
    array = np.frombuffer(raw[: n_frames * frame_size], dtype=np.uint8)
    return array.reshape(n_frames, FRAME_HEIGHT, FRAME_WIDTH).astype(float) / 255.0


def split_segments(segments: list[Segment]) -> tuple[list[Segment], list[Segment]]:
    train = [segment for segment in segments if segment.segment_index % 2 == 0]
    test = [segment for segment in segments if segment.segment_index % 2 == 1]
    return train, test


def evaluate_feature(train: list[Segment], test: list[Segment], feature_fn) -> dict[str, Any]:
    x_train = np.asarray([feature_fn(segment.frames) for segment in train], dtype=float)
    x_test = np.asarray([feature_fn(segment.frames) for segment in test], dtype=float)
    x_train, mean, std = standardise(x_train)
    x_test = (x_test - mean) / std
    labels = sorted({segment.label for segment in train})
    centroids = {
        label: x_train[[segment.label == label for segment in train]].mean(axis=0)
        for label in labels
    }
    predictions = []
    margins = []
    for row in x_test:
        distances = sorted(
            ((float(np.linalg.norm(row - center)), label) for label, center in centroids.items()),
            key=lambda item: (item[0], item[1]),
        )
        predictions.append(distances[0][1])
        margins.append(distances[1][0] - distances[0][0] if len(distances) > 1 else 0.0)
    accuracy = float(np.mean([pred == segment.label for pred, segment in zip(predictions, test)]))
    by_class = {}
    for label in labels:
        selected = [(pred, segment) for pred, segment in zip(predictions, test) if segment.label == label]
        by_class[label] = {
            "num_test_segments": len(selected),
            "accuracy": float(np.mean([pred == segment.label for pred, segment in selected])) if selected else 0.0,
        }
    return {
        "accuracy": accuracy,
        "num_train_segments": len(train),
        "num_test_segments": len(test),
        "mean_nearest_margin": float(np.mean(margins)) if margins else 0.0,
        "by_class": by_class,
    }


def static_appearance_features(frames: np.ndarray) -> np.ndarray:
    mean_frame = frames.mean(axis=0)
    return pooled_grid(mean_frame, rows=6, cols=8)


def passive_next_frame_features(frames: np.ndarray) -> np.ndarray:
    previous = frames[:-1]
    current = frames[1:]
    prediction = previous.mean(axis=0)
    residual = np.abs(current.mean(axis=0) - prediction)
    return np.concatenate([pooled_grid(prediction, 6, 8), pooled_grid(residual, 6, 8)])


def temporal_motion_features(frames: np.ndarray) -> np.ndarray:
    diff = np.abs(np.diff(frames, axis=0))
    quarters = np.array_split(diff, 4, axis=0)
    features = [pooled_grid(part.mean(axis=0), rows=6, cols=8) for part in quarters]
    return np.concatenate(features)


def pooled_grid(image: np.ndarray, rows: int, cols: int) -> np.ndarray:
    row_bins = np.array_split(np.arange(image.shape[0]), rows)
    col_bins = np.array_split(np.arange(image.shape[1]), cols)
    values = []
    for row_idx in row_bins:
        for col_idx in col_bins:
            values.append(float(image[np.ix_(row_idx, col_idx)].mean()))
    return np.asarray(values, dtype=float)


def standardise(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-8] = 1.0
    return (features - mean) / std, mean, std


def _markdown(results: dict[str, Any]) -> str:
    lines = [
        "# KTH Sample Real-Video Benchmark",
        "",
        "This load-bearing benchmark uses official KTH sample AVI files. It is a real-video smoke test, not the full KTH action-recognition benchmark.",
        "",
        "| Learner | Accuracy | Train Segments | Test Segments | Mean Margin |",
        "|---|---:|---:|---:|---:|",
    ]
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
