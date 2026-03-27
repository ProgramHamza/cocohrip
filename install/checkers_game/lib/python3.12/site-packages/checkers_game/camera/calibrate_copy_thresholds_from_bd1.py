"""Calibrate board_detection_copy thresholds using board_detection_1 outputs as reference.

The script compares per-square labels from board_detection_1 (reference) with
variance features extracted from board_detection_copy warped boards, then finds
global thresholds for copy's fallback classifier:

  variance < empty_threshold -> empty (0)
  variance < black_threshold -> black (2)
  else -> white (1)

Usage:
  python -m checkers_game.camera.calibrate_copy_thresholds_from_bd1 --videos-dir ..\\..\\videos_training
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .board_detection_copy import BoardDetection as CopyBoardDetection
from .board_detection_1 import BoardDetection as Bd1BoardDetection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate copy thresholds from bd1 piece labels on copy warp")
    parser.add_argument("--video", type=str, default=None, help="Single video path")
    parser.add_argument("--videos-dir", type=str, default="videos_training", help="Training videos directory")
    parser.add_argument("--max-frames-per-video", type=int, default=300, help="Maximum frames per video")
    parser.add_argument("--frame-step", type=int, default=1, help="Use every Nth frame")
    parser.add_argument("--dark-only", action="store_true", help="Train only on playable dark squares")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON profile path (default: camera/evaluation_output/copy_threshold_calibration.json)",
    )
    parser.add_argument("--display", action="store_true", help="Display debug windows")
    parser.add_argument("--slowmo-ms", type=int, default=1, help="Delay between frames when --display is set")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def resolve_video_path(path_text: str) -> Path:
    raw = Path(path_text).expanduser()
    if raw.exists():
        return raw
    root = repo_root()
    candidates = [root / raw, root / "videos_training" / raw.name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Video not found: {path_text}")


def discover_videos(videos_dir: str) -> list[Path]:
    raw = Path(videos_dir).expanduser()
    base = raw if raw.exists() else (repo_root() / videos_dir)
    if not base.exists() or not base.is_dir():
        return []
    exts = {".mp4", ".mov", ".avi", ".mkv"}
    return sorted([p for p in base.iterdir() if p.is_file() and p.suffix.lower() in exts])


def build_variance_grid(copy_detector: CopyBoardDetection, warped_image: np.ndarray) -> np.ndarray:
    variances = np.full((8, 8), np.nan, dtype=np.float32)
    gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    if not hasattr(copy_detector, "gameBoardFieldsContours") or copy_detector.gameBoardFieldsContours is None:
        copy_detector.gameBoardFieldsContours = copy_detector._get_grid_squares_contours()

    for row in range(8):
        for col in range(8):
            x, y, w_rect, h_rect = copy_detector.gameBoardFieldsContours[row][col]
            padding = 10
            x_pad = x + padding
            y_pad = y + padding
            w_pad = w_rect - 2 * padding
            h_pad = h_rect - 2 * padding

            if w_pad <= 0 or h_pad <= 0:
                continue

            square = blur[y_pad:y_pad + h_pad, x_pad:x_pad + w_pad]
            if square.size == 0:
                continue
            variances[row, col] = float(np.var(square))

    return variances


def predict_from_thresholds(values: np.ndarray, empty_thr: float, black_thr: float) -> np.ndarray:
    preds = np.full_like(values, fill_value=-1, dtype=np.int32)
    valid = np.isfinite(values)
    preds[np.logical_and(valid, values < empty_thr)] = 0
    preds[np.logical_and(valid, values >= empty_thr)] = 2
    preds[np.logical_and(valid, values >= black_thr)] = 1
    return preds


def optimize_thresholds(variances: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    # Sort by variance for O(N) optimal two-threshold split.
    order = np.argsort(variances)
    x = variances[order]
    y = labels[order]
    n = len(x)

    y0 = (y == 0).astype(np.int32)
    y1 = (y == 1).astype(np.int32)
    y2 = (y == 2).astype(np.int32)

    pref0 = np.zeros(n + 1, dtype=np.int64)
    pref1 = np.zeros(n + 1, dtype=np.int64)
    pref2 = np.zeros(n + 1, dtype=np.int64)
    pref0[1:] = np.cumsum(y0)
    pref1[1:] = np.cumsum(y1)
    pref2[1:] = np.cumsum(y2)
    total1 = int(pref1[-1])

    best_score = -1
    best_i = 0
    best_j = 0

    best_prefix_value = -10**18
    best_prefix_index = 0

    for j in range(n + 1):
        candidate_value = int(pref0[j] - pref2[j])
        if candidate_value > best_prefix_value:
            best_prefix_value = candidate_value
            best_prefix_index = j

        i = best_prefix_index
        score = int(pref0[i] + (pref2[j] - pref2[i]) + (total1 - pref1[j]))
        if score > best_score:
            best_score = score
            best_i = i
            best_j = j

    def split_to_threshold(sorted_values: np.ndarray, split_index: int) -> float:
        if split_index <= 0:
            return float(sorted_values[0] - 1e-6)
        if split_index >= len(sorted_values):
            return float(sorted_values[-1] + 1e-6)
        return float((sorted_values[split_index - 1] + sorted_values[split_index]) * 0.5)

    empty_thr = split_to_threshold(x, best_i)
    black_thr = split_to_threshold(x, best_j)
    if black_thr <= empty_thr:
        black_thr = empty_thr + 1.0

    accuracy = float(best_score / n) if n > 0 else 0.0
    return empty_thr, black_thr, accuracy


def confusion_matrix_3(true_labels: np.ndarray, pred_labels: np.ndarray) -> np.ndarray:
    mat = np.zeros((3, 3), dtype=np.int64)
    for t, p in zip(true_labels, pred_labels):
        if 0 <= t <= 2 and 0 <= p <= 2:
            mat[t, p] += 1
    return mat


def calibrate_on_video(video_path: Path, args: argparse.Namespace) -> tuple[list[float], list[int], dict[str, Any]]:
    # copy detector performs one-time corner bootstrap from video
    copy_detector = CopyBoardDetection(
        ximeaCamera=None,
        video_path=str(video_path),
        interactive=False,
        show_windows=args.display,
        loop_video=False,
        prefer_video=True,
    )
    if copy_detector.video_capture is not None:
        copy_detector.video_capture.release()

    bd1_detector = Bd1BoardDetection(ximeaCamera=None)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_idx = 0
    used_frames = 0
    samples_variance: list[float] = []
    samples_label: list[int] = []
    skipped_no_ref = 0
    skipped_no_warp = 0

    original_imshow = cv2.imshow
    if not args.display:
        cv2.imshow = lambda *unused_args, **unused_kwargs: None  # noqa: E731

    try:
        while used_frames < max(1, int(args.max_frames_per_video)):
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if args.frame_step > 1 and (frame_idx % args.frame_step) != 0:
                continue

            if copy_detector.bounderies is None:
                auto = copy_detector._auto_detect_corners_from_image(frame)
                if auto is None:
                    skipped_no_warp += 1
                    continue
                copy_detector.bounderies = auto

            warped = copy_detector._trim_image_perspective(frame, copy_detector.bounderies)
            if warped is None or warped.shape[:2] != (800, 800):
                skipped_no_warp += 1
                continue

            # IMPORTANT: use COPY corners/warp (superior geometry), and use BD1 only
            # for piece classification on the SAME warped image.
            ref_board, _overlay, _black, _white, _empty_thr, _black_thr = bd1_detector._classify_warped_board(warped)
            if ref_board is None:
                skipped_no_ref += 1
                continue
            ref_board_arr = np.array(ref_board, dtype=np.int32)
            if ref_board_arr.shape != (8, 8):
                skipped_no_ref += 1
                continue

            variance_grid = build_variance_grid(copy_detector, warped)
            valid = np.isfinite(variance_grid)
            if args.dark_only:
                dark_mask = np.fromfunction(lambda r, c: ((r + c) % 2 == 1), (8, 8), dtype=int)
                valid = np.logical_and(valid, dark_mask)

            rs, cs = np.where(valid)
            for r, c in zip(rs.tolist(), cs.tolist()):
                samples_variance.append(float(variance_grid[r, c]))
                samples_label.append(int(ref_board_arr[r, c]))

            used_frames += 1

            if args.display:
                key = cv2.waitKey(max(1, int(args.slowmo_ms))) & 0xFF
                if key == 27:
                    break
    finally:
        cap.release()
        cv2.imshow = original_imshow
        if args.display:
            cv2.destroyAllWindows()

    meta = {
        "video_path": str(video_path),
        "reference_mode": "copy_corners_plus_bd1_piece_labels_on_copy_warp",
        "processed_frames": int(used_frames),
        "sample_count": int(len(samples_variance)),
        "skipped_no_reference": int(skipped_no_ref),
        "skipped_no_warp": int(skipped_no_warp),
    }
    return samples_variance, samples_label, meta


def main():
    args = parse_args()
    if args.video:
        videos = [resolve_video_path(args.video)]
    else:
        videos = discover_videos(args.videos_dir)
        if not videos:
            raise SystemExit(f"No videos found in: {args.videos_dir}")

    all_variances: list[float] = []
    all_labels: list[int] = []
    per_video_meta: list[dict[str, Any]] = []

    for idx, video in enumerate(videos, start=1):
        print(f"[{idx}/{len(videos)}] Calibrating from: {video.name}")
        try:
            vars_video, labels_video, meta = calibrate_on_video(video, args)
            all_variances.extend(vars_video)
            all_labels.extend(labels_video)
            per_video_meta.append(meta)
            print(
                f"  frames={meta['processed_frames']} samples={meta['sample_count']} "
                f"skipped_ref={meta['skipped_no_reference']} skipped_warp={meta['skipped_no_warp']}"
            )
        except Exception as exc:
            per_video_meta.append({"video_path": str(video), "error": str(exc)})
            print(f"  error: {exc}")

    if len(all_variances) == 0:
        raise SystemExit("No calibration samples collected.")

    variances_np = np.array(all_variances, dtype=np.float32)
    labels_np = np.array(all_labels, dtype=np.int32)

    # Baseline with copy defaults
    baseline_empty = 15.0
    baseline_black = 1000.0
    baseline_preds = predict_from_thresholds(variances_np, baseline_empty, baseline_black)
    baseline_acc = float(np.mean(baseline_preds == labels_np))

    best_empty, best_black, train_acc = optimize_thresholds(variances_np, labels_np)
    best_preds = predict_from_thresholds(variances_np, best_empty, best_black)
    best_acc = float(np.mean(best_preds == labels_np))

    white_vals = variances_np[labels_np == 1]
    if white_vals.size > 0:
        white_threshold = float(max(best_black, np.percentile(white_vals, 10)))
    else:
        white_threshold = float(best_black)

    conf = confusion_matrix_3(labels_np, best_preds)
    label_counts = {
        "empty_0": int(np.sum(labels_np == 0)),
        "white_1": int(np.sum(labels_np == 1)),
        "black_2": int(np.sum(labels_np == 2)),
    }

    output_path = (
        Path(args.output).expanduser()
        if args.output
        else Path(__file__).resolve().parent / "evaluation_output" / "copy_threshold_calibration.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "empty_variance_threshold": float(best_empty),
        "black_variance_threshold": float(best_black),
        "white_piece_threshold": float(white_threshold),
        "reference_mode": "copy_corners_plus_bd1_piece_labels_on_copy_warp",
        "dark_only": bool(args.dark_only),
        "frame_step": int(args.frame_step),
        "max_frames_per_video": int(args.max_frames_per_video),
        "videos": per_video_meta,
        "sample_count_total": int(len(variances_np)),
        "label_counts": label_counts,
        "baseline_accuracy_copy_defaults": baseline_acc,
        "optimized_accuracy_vs_bd1": best_acc,
        "theoretical_train_accuracy": float(train_acc),
        "confusion_true_rows_pred_cols": conf.tolist(),
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print("\nRecommended thresholds for board_detection_copy:")
    print(f"  empty_variance_threshold = {best_empty:.2f}")
    print(f"  black_variance_threshold = {best_black:.2f}")
    print(f"  white_piece_threshold    = {white_threshold:.2f}")
    print(f"  baseline accuracy        = {baseline_acc * 100:.2f}%")
    print(f"  optimized accuracy       = {best_acc * 100:.2f}%")
    print(f"Saved profile: {output_path}")
    print("To use calibrated thresholds in copy pipeline runtime, initialize with:")
    print("  BoardDetection(..., use_old_board_better_runtime=False)")


if __name__ == "__main__":
    main()
