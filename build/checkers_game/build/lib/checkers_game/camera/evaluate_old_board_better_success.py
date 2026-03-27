"""Evaluate OldBoardBetterDetector success rate on first N frames of each video.

Outputs:
- JSON summary with per-video and per-frame detection results
- Graph: per-video success rate
- Graph: frame-wise success ratio across videos

Usage:
    python -m checkers_game.camera.evaluate_old_board_better_success --frames 60
"""

import argparse
import json
import os
from pathlib import Path
import warnings

import cv2
import matplotlib

warnings.filterwarnings("ignore", message=".*Unable to import Axes3D.*")
os.environ.setdefault("MPLBACKEND", "Agg")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from .old_board_better import (
    OldBoardBetterDetector,
    TemporalBoardStabilizer,
    list_of_videos,
    _resolve_video_path,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark OldBoardBetterDetector success rate")
    parser.add_argument("--frames", type=int, default=60, help="Number of initial frames to evaluate per video")
    parser.add_argument("--hold-frames", type=int, default=3, help="How many missed frames to keep last valid board")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for JSON and graphs")
    return parser.parse_args()


def find_repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def ensure_output_dir(path_arg: str | None) -> Path:
    repo_root = find_repo_root()
    out_dir = Path(path_arg) if path_arg else (repo_root / "src" / "checkers_game" / "checkers_game" / "camera" / "evaluation_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def evaluate_video(video_path: Path, frames_to_check: int, hold_frames: int) -> dict:
    detector = OldBoardBetterDetector()
    stabilizer = TemporalBoardStabilizer(hold_frames=hold_frames)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "video_path": str(video_path),
            "frames_requested": int(frames_to_check),
            "frames_processed": 0,
            "raw_success_count": 0,
            "raw_success_rate": 0.0,
            "raw_frame_success": [],
            "stabilized_success_count": 0,
            "stabilized_success_rate": 0.0,
            "stabilized_frame_success": [],
            "error": "failed_to_open_video",
        }

    raw_frame_success = []
    stabilized_frame_success = []
    frame_idx = 0
    while frame_idx < frames_to_check:
        ok, frame = cap.read()
        if not ok:
            break

        corners, _debug = detector.detect_corners_debug(frame)
        raw_frame_success.append(corners is not None)
        stabilized_corners, _used_hold = stabilizer.update(corners)
        stabilized_frame_success.append(stabilized_corners is not None)
        frame_idx += 1

    cap.release()

    raw_success_count = int(sum(raw_frame_success))
    stabilized_success_count = int(sum(stabilized_frame_success))
    frames_processed = len(raw_frame_success)
    raw_success_rate = float(raw_success_count / frames_processed) if frames_processed > 0 else 0.0
    stabilized_success_rate = float(stabilized_success_count / frames_processed) if frames_processed > 0 else 0.0

    return {
        "video_path": str(video_path),
        "frames_requested": int(frames_to_check),
        "frames_processed": int(frames_processed),
        "raw_success_count": int(raw_success_count),
        "raw_success_rate": raw_success_rate,
        "raw_frame_success": raw_frame_success,
        "stabilized_success_count": int(stabilized_success_count),
        "stabilized_success_rate": stabilized_success_rate,
        "stabilized_frame_success": stabilized_frame_success,
    }


def plot_per_video_success(video_names: list[str], raw_rates: list[float], stabilized_rates: list[float], out_path: Path):
    x = np.arange(len(video_names))
    raw_percentages = [r * 100.0 for r in raw_rates]
    stabilized_percentages = [r * 100.0 for r in stabilized_rates]

    plt.figure(figsize=(10, 5))
    width = 0.38
    bars_raw = plt.bar(x - width / 2, raw_percentages, width=width, color="#42a5f5", label="raw")
    bars_stab = plt.bar(x + width / 2, stabilized_percentages, width=width, color="#66bb6a", label="stabilized")
    plt.xticks(x, video_names, rotation=20, ha="right")
    plt.ylim(0, 100)
    plt.ylabel("Success rate (%)")
    plt.title("OldBoardBetter success rate per video (first N frames): raw vs stabilized")
    plt.legend()

    for i, bar in enumerate(bars_raw):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{raw_percentages[i]:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for i, bar in enumerate(bars_stab):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{stabilized_percentages[i]:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_framewise_success(raw_frame_success_all: list[list[bool]], stabilized_frame_success_all: list[list[bool]], frames_to_check: int, out_path: Path):
    raw_frame_ratios = []
    stabilized_frame_ratios = []
    for idx in range(frames_to_check):
        raw_values = []
        stabilized_values = []
        for video_result in raw_frame_success_all:
            if idx < len(video_result):
                raw_values.append(1.0 if video_result[idx] else 0.0)
        for video_result in stabilized_frame_success_all:
            if idx < len(video_result):
                stabilized_values.append(1.0 if video_result[idx] else 0.0)
        raw_ratio = float(np.mean(raw_values)) if raw_values else 0.0
        stabilized_ratio = float(np.mean(stabilized_values)) if stabilized_values else 0.0
        raw_frame_ratios.append(raw_ratio)
        stabilized_frame_ratios.append(stabilized_ratio)

    x = np.arange(1, frames_to_check + 1)
    y_raw = np.array(raw_frame_ratios) * 100.0
    y_stabilized = np.array(stabilized_frame_ratios) * 100.0

    plt.figure(figsize=(11, 5))
    plt.plot(x, y_raw, color="#42a5f5", linewidth=2, linestyle="--", label="raw")
    plt.plot(x, y_stabilized, color="#66bb6a", linewidth=2, label="stabilized")
    plt.fill_between(x, y_stabilized, alpha=0.15, color="#66bb6a")
    plt.ylim(0, 100)
    plt.xlabel("Frame index (1-based)")
    plt.ylabel("Success across videos (%)")
    plt.title("Frame-wise success ratio (OldBoardBetter): raw vs stabilized")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    frames_to_check = max(1, int(args.frames))
    out_dir = ensure_output_dir(args.output_dir)

    summary = {
        "settings": {
            "frames_per_video": frames_to_check,
            "hold_frames": max(0, int(args.hold_frames)),
            "videos_from_list": list_of_videos,
        },
        "videos": {},
        "overall": {},
    }

    video_names = []
    raw_success_rates = []
    stabilized_success_rates = []
    all_raw_frame_success = []
    all_stabilized_frame_success = []
    total_raw_success = 0
    total_stabilized_success = 0
    total_processed = 0

    for idx, video_name in enumerate(list_of_videos):
        try:
            video_path = _resolve_video_path(video_name)
        except FileNotFoundError as ex:
            summary["videos"][video_name] = {
                "video_path": None,
                "frames_requested": frames_to_check,
                "frames_processed": 0,
                "success_count": 0,
                "success_rate": 0.0,
                "frame_success": [],
                "error": str(ex),
            }
            print(f"[{idx + 1}/{len(list_of_videos)}] {video_name}: missing")
            continue

        result = evaluate_video(video_path, frames_to_check, hold_frames=max(0, int(args.hold_frames)))
        summary["videos"][video_name] = result

        video_names.append(video_name)
        raw_success_rates.append(float(result["raw_success_rate"]))
        stabilized_success_rates.append(float(result["stabilized_success_rate"]))
        all_raw_frame_success.append(result["raw_frame_success"])
        all_stabilized_frame_success.append(result["stabilized_frame_success"])
        total_raw_success += int(result["raw_success_count"])
        total_stabilized_success += int(result["stabilized_success_count"])
        total_processed += int(result["frames_processed"])

        print(
            f"[{idx + 1}/{len(list_of_videos)}] {video_name}: "
            f"raw={result['raw_success_count']}/{result['frames_processed']} ({result['raw_success_rate'] * 100.0:.1f}%), "
            f"stabilized={result['stabilized_success_count']}/{result['frames_processed']} ({result['stabilized_success_rate'] * 100.0:.1f}%)"
        )

    overall_raw_rate = float(total_raw_success / total_processed) if total_processed > 0 else 0.0
    overall_stabilized_rate = float(total_stabilized_success / total_processed) if total_processed > 0 else 0.0
    summary["overall"] = {
        "total_raw_success_count": int(total_raw_success),
        "total_stabilized_success_count": int(total_stabilized_success),
        "total_frames_processed": int(total_processed),
        "overall_raw_success_rate": overall_raw_rate,
        "overall_stabilized_success_rate": overall_stabilized_rate,
    }

    json_path = out_dir / "old_board_better_60f_success_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if video_names:
        per_video_plot = out_dir / "old_board_better_60f_success_per_video.png"
        framewise_plot = out_dir / "old_board_better_60f_success_framewise.png"
        plot_per_video_success(video_names, raw_success_rates, stabilized_success_rates, per_video_plot)
        plot_framewise_success(all_raw_frame_success, all_stabilized_frame_success, frames_to_check, framewise_plot)
        print(f"Saved graph: {per_video_plot}")
        print(f"Saved graph: {framewise_plot}")

    print(f"Saved JSON: {json_path}")
    print(
        "Overall raw: "
        f"{summary['overall']['total_raw_success_count']}/"
        f"{summary['overall']['total_frames_processed']} "
        f"({summary['overall']['overall_raw_success_rate'] * 100.0:.1f}%)"
    )
    print(
        "Overall stabilized: "
        f"{summary['overall']['total_stabilized_success_count']}/"
        f"{summary['overall']['total_frames_processed']} "
        f"({summary['overall']['overall_stabilized_success_rate'] * 100.0:.1f}%)"
    )


if __name__ == "__main__":
    main()
