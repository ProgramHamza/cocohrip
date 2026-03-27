"""Compare old_legacy, old_better, and current detector in parallel debug windows.

Usage:
  python -m checkers_game.camera.offline_compare_old_current --seconds 3 --slowmo-ms 350
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from .grid_corner_detector import GridCornerDetector
from .offline_grid_detect import DEFAULT_VIDEO_PATH, fit_image_to_bounds, make_stage_panel
from .old_board_better import OldBoardBetterDetector
from .old_board_legacy import OldBoardLegacyDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Compare old-legacy vs old-better vs current detector")
    parser.add_argument("--video", type=str, default=DEFAULT_VIDEO_PATH, help="Path to video")
    parser.add_argument("--seconds", type=float, default=3.0, help="Max seconds to process")
    parser.add_argument("--slowmo-ms", type=int, default=350, help="Delay per frame in ms")
    parser.add_argument("--max-panel-width", type=int, default=1300, help="Max width for each panel")
    parser.add_argument("--max-panel-height", type=int, default=850, help="Max height for each panel")
    return parser.parse_args()


def resolve_video(video: str) -> Path:
    p = Path(video).expanduser()
    if p.exists():
        return p
    current = Path(__file__).resolve()
    repo_root = current.parents[4]
    candidates = [repo_root / p, repo_root / "videos_training" / p.name, repo_root / "videos_training" / p.name.replace("_", "-")]
    for c in candidates:
        if c.exists():
            return c
    return p


def to_bgr(img: np.ndarray) -> np.ndarray:
    if img is None:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def make_old_panel(frame: np.ndarray, debug: dict, detected: bool, detector_name: str = "OLD_BETTER") -> np.ndarray:
    base = frame.copy()
    gray = to_bgr(debug.get("gray"))
    clahe = to_bgr(debug.get("clahe"))
    edges = to_bgr(debug.get("edges"))
    closed = to_bgr(debug.get("closed"))
    line_mask = to_bgr(debug.get("line_mask"))
    contour_vis = to_bgr(debug.get("contour_vis"))

    line_overlay = base.copy()
    for x1, y1, x2, y2 in debug.get("lines", []):
        cv2.line(line_overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)

    corners_overlay = base.copy()
    if debug.get("corners") is not None:
        pts = debug["corners"].astype(int)
        for i in range(4):
            cv2.line(corners_overlay, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 0), 2)
            cv2.putText(corners_overlay, str(i), tuple(pts[i]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    top = np.hstack([base, gray, clahe])
    mid = np.hstack([edges, closed, line_overlay])
    bot = np.hstack([line_mask, contour_vis, corners_overlay])
    panel = np.vstack([top, mid, bot])

    status = "DETECTED" if detected else "SEARCHING"
    color = (0, 255, 0) if detected else (0, 165, 255)
    cv2.putText(panel, f"{detector_name} status: {status}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(panel, "Top: original | gray | CLAHE", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(panel, "Mid: edges | closed | Hough lines", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(panel, "Bot: line mask | largest quad candidate | final corners", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(panel, f"raw_lines={len(debug.get('lines', []))}", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    return panel


def main():
    args = parse_args()
    video_path = resolve_video(args.video)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frames = int(fps * args.seconds)

    legacy_detector = OldBoardLegacyDetector()
    old_detector = OldBoardBetterDetector()
    current_detector = GridCornerDetector()

    cv2.namedWindow("old_legacy pipeline", cv2.WINDOW_NORMAL)
    cv2.namedWindow("old_better pipeline", cv2.WINDOW_NORMAL)
    cv2.namedWindow("current pipeline", cv2.WINDOW_NORMAL)

    frame_idx = 0
    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        legacy_corners, legacy_debug = legacy_detector.detect_corners_debug(frame)
        old_corners, old_debug = old_detector.detect_corners_debug(frame)
        cur_result, cur_debug = current_detector.detect_corners_debug(frame)

        legacy_panel = make_old_panel(frame, legacy_debug, legacy_corners is not None, detector_name="OLD_LEGACY")
        old_panel = make_old_panel(frame, old_debug, old_corners is not None, detector_name="OLD_BETTER")
        cur_panel = make_stage_panel(frame, cur_debug, cur_result is not None)

        legacy_panel = fit_image_to_bounds(legacy_panel, args.max_panel_width, args.max_panel_height)
        old_panel = fit_image_to_bounds(old_panel, args.max_panel_width, args.max_panel_height)
        cur_panel = fit_image_to_bounds(cur_panel, args.max_panel_width, args.max_panel_height)

        cv2.putText(legacy_panel, "Detector: OLD_LEGACY (oldboard style)", (20, legacy_panel.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(old_panel, "Detector: OLD_BETTER (improved old)", (20, old_panel.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(cur_panel, "Detector: CURRENT", (20, cur_panel.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("old_legacy pipeline", legacy_panel)
        cv2.imshow("old_better pipeline", old_panel)
        cv2.imshow("current pipeline", cur_panel)

        key = cv2.waitKey(max(1, args.slowmo_ms)) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
