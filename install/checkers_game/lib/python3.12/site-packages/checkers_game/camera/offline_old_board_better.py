"""
Display only the OldBoardBetter pipeline with debug 3x3 window for a given video.
Usage:
  python -m checkers_game.camera.offline_old_board_better --video path/to/video.mp4 --seconds 9999 --slowmo-ms 0
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from .old_board_better import OldBoardBetterDetector
from .offline_grid_detect import fit_image_to_bounds, make_stage_panel

def parse_args():
    parser = argparse.ArgumentParser(description="Show OldBoardBetter pipeline only")
    parser.add_argument("--video", type=str, required=True, help="Path to video")
    parser.add_argument("--seconds", type=float, default=9999, help="Max seconds to process")
    parser.add_argument("--slowmo-ms", type=int, default=0, help="Delay per frame in ms")
    parser.add_argument("--max-panel-width", type=int, default=1300, help="Max width for panel")
    parser.add_argument("--max-panel-height", type=int, default=850, help="Max height for panel")
    return parser.parse_args()

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
    video_path = Path(args.video).expanduser()
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frames = int(fps * args.seconds)

    old_detector = OldBoardBetterDetector()
    cv2.namedWindow("old_better pipeline", cv2.WINDOW_NORMAL)

    frame_idx = 0
    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        old_corners, old_debug = old_detector.detect_corners_debug(frame)
        old_panel = make_old_panel(frame, old_debug, old_corners is not None, detector_name="OLD_BETTER")
        old_panel = fit_image_to_bounds(old_panel, args.max_panel_width, args.max_panel_height)
        cv2.putText(old_panel, "Detector: OLD_BETTER (improved old)", (20, old_panel.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("old_better pipeline", old_panel)

        key = cv2.waitKey(max(1, args.slowmo_ms)) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
