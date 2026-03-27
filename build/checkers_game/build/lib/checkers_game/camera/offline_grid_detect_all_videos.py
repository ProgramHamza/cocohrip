"""Run grid detection debug view on all videos in videos_training one by one.

This script does NOT change detection logic; it only iterates videos and visualizes.

Usage:
    python -m checkers_game.camera.offline_grid_detect_all_videos --seconds 3 --slowmo-ms 350
"""

import argparse
from pathlib import Path

import cv2

from .grid_corner_detector import GridCornerDetector, warp_board
from .offline_grid_detect import draw_debug, fit_image_to_bounds, make_stage_panel


def parse_args():
    parser = argparse.ArgumentParser(description="Run debug detection on all training videos")
    parser.add_argument("--seconds", type=float, default=3.0, help="Max seconds per video")
    parser.add_argument("--slowmo-ms", type=int, default=350, help="Delay per frame in ms")
    parser.add_argument("--max-panel-width", type=int, default=1400, help="Max debug panel width")
    parser.add_argument("--max-panel-height", type=int, default=900, help="Max debug panel height")
    return parser.parse_args()


def find_videos():
    current = Path(__file__).resolve()
    # /home/.../amavet_ws/src/checkers_game/checkers_game/camera/file.py -> repo root is parents[4]
    repo_root = current.parents[4]

    candidates = [
        repo_root / "videos_training",
        repo_root / "src" / "videos_training",
        Path.cwd() / "videos_training",
        Path.cwd().parent / "videos_training",
        Path.cwd().parent.parent / "videos_training",
    ]

    videos = []
    for videos_dir in candidates:
        if not videos_dir.exists() or not videos_dir.is_dir():
            continue
        for ext in ("*.mp4", "*.mov", "*.avi", "*.mkv"):
            videos.extend(videos_dir.glob(ext))
        if videos:
            break

    return sorted(set(videos))


def main():
    args = parse_args()
    videos = find_videos()
    if not videos:
        raise SystemExit("No videos found in videos_training/")

    detector = GridCornerDetector()

    cv2.namedWindow("grid detection pipeline", cv2.WINDOW_NORMAL)

    for idx, video_path in enumerate(videos, start=1):
        print(f"\n[{idx}/{len(videos)}] Processing: {video_path.name}")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  -> Failed to open, skipping: {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        max_frames = int(fps * args.seconds)
        frame_idx = 0
        detected = None

        skip_video = False
        stop_all = False

        while frame_idx < max_frames:
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1
            result, debug = detector.detect_corners_debug(frame)

            panel = make_stage_panel(frame, debug, result is not None)
            panel = fit_image_to_bounds(panel, args.max_panel_width, args.max_panel_height)

            cv2.putText(panel, f"Video {idx}/{len(videos)}: {video_path.name}", (20, panel.shape[0] - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(panel, "ESC=quit all, N=next video", (20, panel.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("grid detection pipeline", panel)
            key = cv2.waitKey(max(1, args.slowmo_ms)) & 0xFF

            if key == 27:  # ESC
                stop_all = True
                break
            if key == ord('n') or key == ord('N'):
                skip_video = True
                break

            if result is not None:
                detected = (frame, result)
                break

        cap.release()

        if stop_all:
            break

        if skip_video:
            print("  -> Skipped by user")
            continue

        if detected is None:
            print("  -> No grid detected within time budget")
            continue

        frame, grid = detected
        warped = warp_board(frame, grid.corners)
        debug_overlay = draw_debug(frame, grid.corners, grid.intersections)
        cv2.imshow("original + grid", debug_overlay)
        cv2.imshow("warped board", warped)

        print("  -> Grid detected. Press any key for next video (ESC to quit all).")
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow("original + grid")
        cv2.destroyWindow("warped board")
        if key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
