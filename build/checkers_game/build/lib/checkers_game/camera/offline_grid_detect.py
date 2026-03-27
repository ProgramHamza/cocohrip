"""Offline grid corner detection on recorded videos.

Usage (from workspace root):
    python -m checkers_game.camera.offline_grid_detect --video videos_training/sample.mp4
"""

import argparse
import glob
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .grid_corner_detector import GridCornerDetector, warp_board


DEFAULT_VIDEO_PATH = "/home/marek/amavet_ws/videos_training/2026-02-25_17-27-27.mp4"


def parse_args():
    parser = argparse.ArgumentParser(description="Detect board corners from recorded video")
    parser.add_argument("--video", type=str, default=DEFAULT_VIDEO_PATH, help="Path to input video")
    parser.add_argument("--seconds", type=float, default=3.0, help="Max seconds to scan before giving up")
    parser.add_argument("--display", action="store_true", help="Show debug windows")
    parser.add_argument("--slowmo-ms", type=int, default=180, help="Delay per frame in ms when --display is enabled")
    parser.add_argument("--max-panel-width", type=int, default=1400, help="Max debug panel width in pixels")
    parser.add_argument("--max-panel-height", type=int, default=900, help="Max debug panel height in pixels")
    parser.add_argument("--parallel-tol", type=float, default=18.0, help="Base angular tolerance for parallel line grouping (deg)")
    parser.add_argument("--spacing-tol", type=float, default=0.90, help="Allowed std/mean ratio for line spacing consistency")
    parser.add_argument("--merge-angle-tol", type=float, default=4.0, help="Angular tolerance for merging short collinear segments (deg)")
    parser.add_argument("--merge-dist-px", type=float, default=6.0, help="Distance tolerance in px for merging collinear segments")
    parser.add_argument("--thickness-threshold", type=float, default=0.36, help="Contour-thickness periodicity threshold for keeping a line")
    parser.add_argument("--thickness-min-keep", type=int, default=18, help="Minimum number of lines to keep after thickness filtering fallback")
    parser.add_argument("--outline-mod-threshold", type=float, default=0.40, help="Threshold for contour-thickness modulation outline detector")
    parser.add_argument("--outline-min-length-ratio", type=float, default=0.22, help="Minimum line length ratio for outline candidate lines")
    return parser.parse_args()


def find_default_video() -> Optional[Path]:
    """Pick the first mp4/mov in ../../videos_training relative to repo root."""
    current = Path(__file__).resolve()
    # repo root (/home/.../amavet_ws) is five levels up from this file
    repo_root = current.parents[5]
    candidates = list(repo_root.glob("videos_training/*.*"))
    videos = [p for p in candidates if p.suffix.lower() in {".mp4", ".mov", ".avi"}]
    return videos[0] if videos else None


def draw_debug(frame: np.ndarray, corners: np.ndarray, intersections):
    vis = frame.copy()
    for x, y in intersections:
        cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 255), -1)
    pts = corners.reshape(-1, 2).astype(int)
    for i in range(4):
        cv2.line(vis, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 0), 2)
    for idx, (x, y) in enumerate(pts):
        cv2.putText(vis, str(idx), (int(x) + 4, int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return vis


def to_bgr(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def make_stage_panel(frame: np.ndarray, debug: dict, detected: bool) -> np.ndarray:
    base = frame.copy()
    lines_overlay = base.copy()
    v_lines_overlay = base.copy()
    i_overlay = base.copy()

    for x1, y1, x2, y2 in debug.get("all_lines", []):
        cv2.line(lines_overlay, (x1, y1), (x2, y2), (255, 255, 0), 1)

    merged_overlay = base.copy()
    for x1, y1, x2, y2 in debug.get("merged_lines", []):
        cv2.line(merged_overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)

    contour_overlay = to_bgr(debug.get("contour_binary", np.zeros(frame.shape[:2], dtype=np.uint8)))
    for x1, y1, x2, y2 in debug.get("outline_lines", []):
        cv2.line(contour_overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)

    for x1, y1, x2, y2 in debug.get("vertical_lines", []):
        cv2.line(v_lines_overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for x1, y1, x2, y2 in debug.get("horizontal_lines", []):
        cv2.line(v_lines_overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)

    for x, y in debug.get("intersections", []):
        cv2.circle(i_overlay, (int(x), int(y)), 2, (0, 255, 255), -1)

    if debug.get("corners") is not None:
        pts = debug["corners"].astype(int)
        for idx, pt in enumerate(pts):
            cv2.circle(i_overlay, tuple(pt), 6, (0, 255, 0), -1)
            cv2.putText(i_overlay, str(idx), (pt[0] + 5, pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        for idx in range(4):
            cv2.line(i_overlay, tuple(pts[idx]), tuple(pts[(idx + 1) % 4]), (0, 255, 0), 2)

    compare_overlay = base.copy()
    if debug.get("legacy_corners") is not None:
        legacy_pts = debug["legacy_corners"].astype(int)
        for idx in range(4):
            cv2.line(compare_overlay, tuple(legacy_pts[idx]), tuple(legacy_pts[(idx + 1) % 4]), (0, 165, 255), 2)
    if debug.get("corners") is not None:
        cur_pts = debug["corners"].astype(int)
        for idx in range(4):
            cv2.line(compare_overlay, tuple(cur_pts[idx]), tuple(cur_pts[(idx + 1) % 4]), (0, 255, 0), 2)

    gray = to_bgr(debug["gray"])
    clahe = to_bgr(debug["clahe"])
    edges = to_bgr(debug["edges"])
    closed = to_bgr(debug["closed"])
    closed_with_raw = closed.copy()
    for x1, y1, x2, y2 in debug.get("all_lines", []):
        cv2.line(closed_with_raw, (x1, y1), (x2, y2), (0, 255, 255), 2)

    top = np.hstack([base, gray, clahe])
    mid = np.hstack([edges, closed_with_raw, contour_overlay])
    bot = np.hstack([v_lines_overlay, i_overlay, compare_overlay])
    panel = np.vstack([top, mid, bot])

    status = "DETECTED" if detected else "SEARCHING"
    color = (0, 255, 0) if detected else (0, 165, 255)
    cv2.putText(panel, f"Pipeline status: {status}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(panel, "Top: original | gray | contrast(CLAHE)", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(panel, "Mid: edges | morphology + extracted lines (5th) | contour-map + outline", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(panel, "Bot: clustered lines | intersections+corners | current(green) vs legacy(orange)", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    total_lines = len(debug.get("all_lines", []))
    merged_lines = len(debug.get("merged_lines", []))
    thickness_pass = int(debug.get("thickness_pass_count", 0))
    outline_lines = len(debug.get("outline_lines", []))
    v_count = len(debug.get("vertical_lines", []))
    h_count = len(debug.get("horizontal_lines", []))
    intersections = len(debug.get("intersections", []))
    cv2.putText(panel, f"raw={total_lines} merged={merged_lines} thickness_pass={thickness_pass} outline={outline_lines} familyA={v_count} familyB={h_count} inter={intersections}",
                (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

    stats = debug.get("family_stats", {})
    source = debug.get("selected_source", "current")
    cur_score = debug.get("current_score", None)
    leg_score = debug.get("legacy_score", None)
    agree = debug.get("corner_agreement", None)
    outline_scores = debug.get("outline_scores", [])
    a = stats.get("family_a", {})
    b = stats.get("family_b", {})
    a_in = int(a.get("input_count", 0))
    b_in = int(b.get("input_count", 0))
    a_par = int(a.get("parallel_count", 0))
    b_par = int(b.get("parallel_count", 0))
    a_dd = int(a.get("dedup_count", 0))
    b_dd = int(b.get("dedup_count", 0))
    a_gap = a.get("mean_gap", -1.0)
    b_gap = b.get("mean_gap", -1.0)
    a_std = a.get("std_gap", -1.0)
    b_std = b.get("std_gap", -1.0)
    a_tol = a.get("adaptive_tol", -1.0)
    b_tol = b.get("adaptive_tol", -1.0)
    cv2.putText(panel, f"A in/par/dedup/sel: {a_in}/{a_par}/{a_dd}/{v_count}   B in/par/dedup/sel: {b_in}/{b_par}/{b_dd}/{h_count}",
                (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    cv2.putText(panel, f"A gap mean/std: {a_gap:.2f}/{a_std:.2f}   B gap mean/std: {b_gap:.2f}/{b_std:.2f}",
                (20, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    cv2.putText(panel, f"Adaptive angle tol A/B: {a_tol:.1f}/{b_tol:.1f} deg", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    if outline_scores:
        top_scores = sorted([float(s) for s in outline_scores], reverse=True)[:4]
        cv2.putText(panel, f"Outline category sizes: {top_scores}", (20, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    cur_txt = f"{cur_score:.3f}" if isinstance(cur_score, (int, float)) else "n/a"
    leg_txt = f"{leg_score:.3f}" if isinstance(leg_score, (int, float)) else "n/a"
    agr_txt = f"{agree:.3f}" if isinstance(agree, (int, float)) else "n/a"
    cv2.putText(panel, f"Source={source}  current_score={cur_txt}  legacy_score={leg_txt}  agree={agr_txt}",
                (20, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    return panel


def fit_image_to_bounds(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= max_width and h <= max_height:
        return image
    scale = min(max_width / float(w), max_height / float(h))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def main():
    args = parse_args()
    video_path = Path(args.video).expanduser() if args.video else Path(DEFAULT_VIDEO_PATH)

    if video_path is not None and not video_path.exists():
        # Try resolving relative to repo root/videos_training and a dashed-name variant
        current = Path(__file__).resolve()
        repo_root = current.parents[5]

        # Try provided relative to repo root
        alt_candidates = [repo_root / video_path]

        # Try videos_training with same name
        alt_candidates.append(repo_root / "videos_training" / video_path.name)

        # Try a dashed variant (underscores to dashes) for the basename
        dashed_name = video_path.name.replace("_", "-")
        alt_candidates.append(repo_root / "videos_training" / dashed_name)

        video_path = next((p for p in alt_candidates if p.exists()), video_path)

    if video_path is None or not video_path.exists():
        raise SystemExit(f"No video found. Tried: {video_path if video_path is not None else 'None'}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frames = int(fps * args.seconds)

    detector = GridCornerDetector()
    detector.parallel_angle_tolerance_deg = max(5.0, float(args.parallel_tol))
    detector.spacing_tolerance_ratio = max(0.1, float(args.spacing_tol))
    detector.merge_angle_tolerance_deg = max(0.5, float(args.merge_angle_tol))
    detector.merge_distance_tolerance_px = max(1.0, float(args.merge_dist_px))
    detector.thickness_score_threshold = max(0.01, float(args.thickness_threshold))
    detector.thickness_min_keep = max(6, int(args.thickness_min_keep))
    detector.outline_modulation_threshold = max(0.05, float(args.outline_mod_threshold))
    detector.outline_min_length_ratio = max(0.05, min(0.8, float(args.outline_min_length_ratio)))
    detected = None
    frame_idx = 0

    if args.display:
        cv2.namedWindow("grid detection pipeline", cv2.WINDOW_NORMAL)

    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        result, debug = detector.detect_corners_debug(frame)

        if args.display:
            panel = make_stage_panel(frame, debug, result is not None)
            panel = fit_image_to_bounds(panel, args.max_panel_width, args.max_panel_height)
            cv2.imshow("grid detection pipeline", panel)
            key = cv2.waitKey(max(1, args.slowmo_ms)) & 0xFF
            if key == 27:
                break

        if result is not None:
            detected = (frame, result)
            print(f"Detected grid on frame {frame_idx}/{max_frames} (t={frame_idx/fps:.2f}s)")
            break

    cap.release()

    if detected is None:
        print("No grid detected within time budget")
        if args.display:
            cv2.destroyAllWindows()
        return

    frame, grid = detected
    warped = warp_board(frame, grid.corners)

    print("Corners (TL, TR, BR, BL):")
    for i, (x, y) in enumerate(grid.corners):
        print(f"  {i}: ({x:.1f}, {y:.1f})")

    if args.display:
        debug = draw_debug(frame, grid.corners, grid.intersections)
        cv2.imshow("original + grid", debug)
        cv2.imshow("warped board", warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Hook point: pass `warped` into existing board/piece tracker once thresholds are known.


if __name__ == "__main__":
    main()
