"""Evaluate detector corner accuracy against manually labeled ground truth.

Outputs:
1) Bar chart: correct detections out of N for each video x method (6 bars for 3 videos x 2 methods).
2) Wrongness chart: average corner error for wrong detections (same groups).
3) JSON summary with per-video metrics.
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

from .grid_corner_detector import GridCornerDetector
from .old_board_better import OldBoardBetterDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate corner detectors against ground truth labels")
    parser.add_argument("--labels", type=str, default=None, help="Path to ground_truth_corners.json")
    parser.add_argument("--seconds", type=float, default=3.0, help="Seconds to scan in each video")
    parser.add_argument("--target-detections", type=int, default=10, help="Number of detections to collect per method per video")
    parser.add_argument("--correct-threshold", type=float, default=20.0, help="Mean corner error threshold (px) to count as correct")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for graphs and summary JSON")
    return parser.parse_args()


def find_repo_root():
    return Path(__file__).resolve().parents[4]


def resolve_labels_path(path_arg):
    repo_root = find_repo_root()
    default_path = repo_root / "src" / "checkers_game" / "checkers_game" / "camera" / "ground_truth_corners.json"
    return Path(path_arg) if path_arg else default_path


def find_videos():
    repo_root = find_repo_root()
    candidates = [
        repo_root / "videos_training",
        repo_root / "src" / "videos_training",
        Path.cwd() / "videos_training",
        Path.cwd().parent / "videos_training",
        Path.cwd().parent.parent / "videos_training",
    ]

    videos = []
    for folder in candidates:
        if not folder.exists() or not folder.is_dir():
            continue
        for ext in ("*.mp4", "*.mov", "*.avi", "*.mkv"):
            videos.extend(folder.glob(ext))
        if videos:
            break
    return {p.name: p for p in sorted(set(videos))}


def mean_corner_error(pred, gt):
    pred = np.array(pred, dtype=np.float32).reshape(4, 2)
    gt = np.array(gt, dtype=np.float32).reshape(4, 2)
    return float(np.mean(np.linalg.norm(pred - gt, axis=1)))


def collect_detections(video_path: Path, seconds: float, target: int):
    old_detector = OldBoardBetterDetector()
    cur_detector = GridCornerDetector()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frames = int(max(1.0, seconds) * fps)

    old_list = []
    cur_list = []
    frame_idx = 0

    while frame_idx < max_frames and (len(old_list) < target or len(cur_list) < target):
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        if len(old_list) < target:
            old_corners, _old_debug = old_detector.detect_corners_debug(frame)
            if old_corners is not None:
                old_list.append(np.array(old_corners, dtype=np.float32).reshape(4, 2))

        if len(cur_list) < target:
            cur_result, _cur_debug = cur_detector.detect_corners_debug(frame)
            if cur_result is not None:
                cur_list.append(np.array(cur_result.corners, dtype=np.float32).reshape(4, 2))

    cap.release()
    return old_list, cur_list


def evaluate_method(detections, gt_corners, correct_threshold):
    errors = [mean_corner_error(c, gt_corners) for c in detections]
    correct = [e for e in errors if e <= correct_threshold]
    wrong = [e for e in errors if e > correct_threshold]
    return {
        "total": len(errors),
        "correct_count": len(correct),
        "wrong_count": len(wrong),
        "errors": errors,
        "wrong_errors": wrong,
        "mean_error": float(np.mean(errors)) if errors else None,
        "mean_wrong_error": float(np.mean(wrong)) if wrong else None,
    }


def ensure_output_dir(path_arg):
    repo_root = find_repo_root()
    out = Path(path_arg) if path_arg else (repo_root / "src" / "checkers_game" / "checkers_game" / "camera" / "evaluation_output")
    out.mkdir(parents=True, exist_ok=True)
    return out


def plot_correct_counts(records, out_path):
    labels = [r["label"] for r in records]
    values = [r["correct_count"] for r in records]
    totals = [r["total"] for r in records]

    x = np.arange(len(labels))
    plt.figure(figsize=(11, 5))
    bars = plt.bar(x, values, color="#4caf50")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Correct detections")
    plt.title("Correct detections out of N (video × method)")
    for i, b in enumerate(bars):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1, f"{values[i]}/{totals[i]}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_wrongness(records, out_path):
    labels = [r["label"] for r in records]
    values = [r["mean_wrong_error"] if r["mean_wrong_error"] is not None else 0.0 for r in records]

    x = np.arange(len(labels))
    plt.figure(figsize=(11, 5))
    bars = plt.bar(x, values, color="#ff7043")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Mean wrong corner error (px)")
    plt.title("How wrong were wrong detections (video × method)")
    for i, b in enumerate(bars):
        txt = "n/a" if values[i] == 0.0 and records[i]["wrong_count"] == 0 else f"{values[i]:.1f}px"
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1, txt, ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    args = parse_args()

    labels_path = resolve_labels_path(args.labels)
    if not labels_path.exists():
        raise SystemExit(f"Labels file not found: {labels_path}")

    with labels_path.open("r", encoding="utf-8") as f:
        labels = json.load(f)

    videos = find_videos()
    out_dir = ensure_output_dir(args.output_dir)

    summary = {
        "settings": {
            "seconds": args.seconds,
            "target_detections": args.target_detections,
            "correct_threshold": args.correct_threshold,
        },
        "videos": {},
    }

    graph_records = []

    total_labeled = len(labels)
    processed = 0

    for video_name, gt_item in labels.items():
        if video_name not in videos:
            print(f"Skip (video not found): {video_name}")
            continue

        processed += 1
        print(f"[{processed}/{total_labeled}] Evaluating: {video_name}")

        gt_corners = np.array(gt_item["corners"], dtype=np.float32).reshape(4, 2)
        old_detections, cur_detections = collect_detections(videos[video_name], args.seconds, args.target_detections)

        old_eval = evaluate_method(old_detections, gt_corners, args.correct_threshold)
        cur_eval = evaluate_method(cur_detections, gt_corners, args.correct_threshold)

        summary["videos"][video_name] = {
            "old_better": old_eval,
            "current": cur_eval,
        }

        graph_records.append(
            {
                "label": f"{video_name}\nold",
                "correct_count": old_eval["correct_count"],
                "total": old_eval["total"],
                "wrong_count": old_eval["wrong_count"],
                "mean_wrong_error": old_eval["mean_wrong_error"],
            }
        )
        graph_records.append(
            {
                "label": f"{video_name}\ncurrent",
                "correct_count": cur_eval["correct_count"],
                "total": cur_eval["total"],
                "wrong_count": cur_eval["wrong_count"],
                "mean_wrong_error": cur_eval["mean_wrong_error"],
            }
        )

        print(
            f"{video_name}: old={old_eval['correct_count']}/{old_eval['total']} correct, "
            f"current={cur_eval['correct_count']}/{cur_eval['total']} correct"
        )

    if not graph_records:
        raise SystemExit("No labeled videos were evaluated.")

    correct_graph = out_dir / "correct_out_of_n.png"
    wrong_graph = out_dir / "wrongness_mean_error.png"
    plot_correct_counts(graph_records, correct_graph)
    plot_wrongness(graph_records, wrong_graph)

    summary_path = out_dir / "evaluation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Evaluation finished.")
    print(f"Saved summary: {summary_path}")
    print(f"Saved graph 1 (correct out of N): {correct_graph}")
    print(f"Saved graph 2 (how wrong wrong detections were): {wrong_graph}")


if __name__ == "__main__":
    main()
