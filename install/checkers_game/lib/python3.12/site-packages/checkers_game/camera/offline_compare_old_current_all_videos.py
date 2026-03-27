"""Compare old_legacy vs old_better vs current detector across all videos one by one.

Usage:
  PYTHONPATH=/home/marek/amavet_ws/src/checkers_game \
  python3 -m checkers_game.camera.offline_compare_old_current_all_videos --seconds 3 --slowmo-ms 350
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from .grid_corner_detector import GridCornerDetector, warp_board
from .offline_compare_old_current import make_old_panel
from .offline_grid_detect import draw_debug, fit_image_to_bounds, make_stage_panel
from .old_board_better import OldBoardBetterDetector
from .old_board_legacy import OldBoardLegacyDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Compare old_legacy, old_better, and current detector on all videos")
    parser.add_argument("--seconds", type=float, default=3.0, help="Max seconds per video")
    parser.add_argument("--slowmo-ms", type=int, default=350, help="Delay per frame in ms")
    parser.add_argument("--max-panel-width", type=int, default=1300, help="Max width for each debug window")
    parser.add_argument("--max-panel-height", type=int, default=850, help="Max height for each debug window")
    parser.add_argument("--freeze-on-detect", action="store_true",
                        help="Freeze current video immediately when first detection appears")
    parser.add_argument("--target-detections", type=int, default=10,
                        help="Collect up to this many detections per method before selecting most frequent result")
    parser.add_argument("--cluster-threshold", type=float, default=18.0,
                        help="Corner clustering threshold (px) for frequency grouping")
    return parser.parse_args()


def find_videos():
    current = Path(__file__).resolve()
    repo_root = current.parents[4]

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
    return sorted(set(videos))


def main():
    args = parse_args()
    videos = find_videos()
    if not videos:
        raise SystemExit("No videos found in videos_training/")

    legacy_detector = OldBoardLegacyDetector()
    old_detector = OldBoardBetterDetector()
    current_detector = GridCornerDetector()

    cv2.namedWindow("old_legacy pipeline", cv2.WINDOW_NORMAL)
    cv2.namedWindow("old_better pipeline", cv2.WINDOW_NORMAL)
    cv2.namedWindow("current pipeline", cv2.WINDOW_NORMAL)

    overall_summary = []

    def choose_most_frequent(detections, cluster_threshold):
        """detections: list of tuples (score, frame, corners). Returns representative tuple and stats."""
        if not detections:
            return None, {"clusters": 0, "best_count": 0, "total": 0}

        clusters = []  # each: {"rep": np.ndarray, "items": [...]}
        for item in detections:
            _score, _frame, corners = item
            c = np.array(corners, dtype=np.float32).reshape(4, 2)
            placed = False
            for cl in clusters:
                rep = cl["rep"]
                d = float(np.mean(np.linalg.norm(c - rep, axis=1)))
                if d <= cluster_threshold:
                    cl["items"].append(item)
                    # update representative as running mean of corners
                    arr = np.array([np.array(it[2], dtype=np.float32).reshape(4, 2) for it in cl["items"]], dtype=np.float32)
                    cl["rep"] = np.mean(arr, axis=0)
                    placed = True
                    break
            if not placed:
                clusters.append({"rep": c, "items": [item]})

        # Best cluster by frequency, then by avg score
        best_cluster = None
        best_key = None
        for cl in clusters:
            count = len(cl["items"])
            avg_score = float(np.mean([it[0] for it in cl["items"]]))
            key = (count, avg_score)
            if best_key is None or key > best_key:
                best_key = key
                best_cluster = cl

        best_item = max(best_cluster["items"], key=lambda x: x[0])
        stats = {"clusters": len(clusters), "best_count": len(best_cluster["items"]), "total": len(detections)}
        return best_item, stats

    for idx, video_path in enumerate(videos, start=1):
        print(f"\n[{idx}/{len(videos)}] Processing: {video_path.name}")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  -> Failed to open, skipping: {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        max_frames = int(fps * args.seconds)
        frame_idx = 0

        legacy_detected = None
        old_detected = None
        cur_detected = None
        legacy_scores = []
        old_scores = []
        cur_scores = []
        legacy_corners_seq = []
        old_corners_seq = []
        cur_corners_seq = []
        skip_video = False
        stop_all = False

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

            cv2.putText(legacy_panel, f"Video {idx}/{len(videos)}: {video_path.name}", (20, legacy_panel.shape[0] - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(legacy_panel, "ESC=quit all, N=next video", (20, legacy_panel.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.putText(old_panel, f"Video {idx}/{len(videos)}: {video_path.name}", (20, old_panel.shape[0] - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(old_panel, "ESC=quit all, N=next video", (20, old_panel.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.putText(cur_panel, f"Video {idx}/{len(videos)}: {video_path.name}", (20, cur_panel.shape[0] - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(cur_panel, "ESC=quit all, N=next video", (20, cur_panel.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("old_legacy pipeline", legacy_panel)
            cv2.imshow("old_better pipeline", old_panel)
            cv2.imshow("current pipeline", cur_panel)

            key = cv2.waitKey(max(1, args.slowmo_ms)) & 0xFF
            if key == 27:
                stop_all = True
                break
            if key == ord('n') or key == ord('N'):
                skip_video = True
                break

            if legacy_corners is not None and legacy_detected is None:
                legacy_detected = (frame.copy(), legacy_corners)
            if legacy_corners is not None:
                score_legacy = float(current_detector._checkerboard_region_score(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), legacy_corners))
                legacy_scores.append((score_legacy, frame.copy(), legacy_corners.copy()))
                legacy_corners_seq.append(legacy_corners.copy())

            if old_corners is not None and old_detected is None:
                old_detected = (frame.copy(), old_corners)
            if old_corners is not None:
                score_old = float(current_detector._checkerboard_region_score(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), old_corners))
                old_scores.append((score_old, frame.copy(), old_corners.copy()))
                old_corners_seq.append(old_corners.copy())

            if cur_result is not None and cur_detected is None:
                cur_detected = (frame.copy(), cur_result)
            if cur_result is not None:
                if isinstance(cur_debug.get("current_score", None), (float, int)):
                    score_cur = float(cur_debug["current_score"])
                else:
                    score_cur = float(current_detector._checkerboard_region_score(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cur_result.corners))
                cur_scores.append((score_cur, frame.copy(), cur_result))
                cur_corners_seq.append(cur_result.corners.copy())

            # Optional freeze on first detection.
            if args.freeze_on_detect and (legacy_corners is not None or old_corners is not None or cur_result is not None):
                break

            # Default: keep running until we collect enough detections (up to target) for frequency mode.
            if len(legacy_scores) >= args.target_detections and len(old_scores) >= args.target_detections and len(cur_scores) >= args.target_detections:
                break

        cap.release()

        if stop_all:
            break
        if skip_video:
            print("  -> Skipped by user")
            continue

        # Build concrete per-method outcome by frequency (no winner selection).
        total_frames = max(1, frame_idx)
        legacy_hit_rate = len(legacy_scores) / total_frames
        old_hit_rate = len(old_scores) / total_frames
        cur_hit_rate = len(cur_scores) / total_frames

        def top_mean(scores, k=3):
            if not scores:
                return -1e6
            vals = sorted([s[0] for s in scores], reverse=True)[:k]
            return float(np.mean(vals))

        def jitter(corners_seq):
            if len(corners_seq) < 2:
                return 0.0
            arr = np.array(corners_seq, dtype=np.float32)
            diffs = np.linalg.norm(arr[1:] - arr[:-1], axis=2)
            return float(np.mean(diffs))

        legacy_top = top_mean(legacy_scores)
        old_top = top_mean(old_scores)
        cur_top = top_mean(cur_scores)
        legacy_jitter = jitter(legacy_corners_seq)
        old_jitter = jitter(old_corners_seq)
        cur_jitter = jitter(cur_corners_seq)

        legacy_metric = legacy_top + 0.8 * legacy_hit_rate - 0.02 * legacy_jitter
        old_metric = old_top + 0.8 * old_hit_rate - 0.02 * old_jitter
        cur_metric = cur_top + 0.8 * cur_hit_rate - 0.02 * cur_jitter

        legacy_rep, legacy_freq = choose_most_frequent(legacy_scores, args.cluster_threshold)
        old_rep, old_freq = choose_most_frequent(old_scores, args.cluster_threshold)
        cur_rep, cur_freq = choose_most_frequent(
            [(s, f, r.corners.copy()) for (s, f, r) in cur_scores],
            args.cluster_threshold,
        )

        print(
            f"  -> Outcome: "
            f"old_legacy(hit={legacy_hit_rate:.2f}, top={legacy_top:.3f}, jitter={legacy_jitter:.2f}, metric={legacy_metric:.3f}, "
            f"freq={legacy_freq['best_count']}/{legacy_freq['total']}, clusters={legacy_freq['clusters']}) | "
            f"current(hit={cur_hit_rate:.2f}, top={cur_top:.3f}, jitter={cur_jitter:.2f}, metric={cur_metric:.3f}, "
            f"freq={cur_freq['best_count']}/{cur_freq['total']}, clusters={cur_freq['clusters']}) | "
            f"old(hit={old_hit_rate:.2f}, top={old_top:.3f}, jitter={old_jitter:.2f}, metric={old_metric:.3f}, "
            f"freq={old_freq['best_count']}/{old_freq['total']}, clusters={old_freq['clusters']})"
        )

        overall_summary.append(
            {
                "video": video_path.name,
                "legacy_hit": legacy_hit_rate,
                "current_hit": cur_hit_rate,
                "old_hit": old_hit_rate,
                "legacy_metric": legacy_metric,
                "current_metric": cur_metric,
                "old_metric": old_metric,
                "legacy_freq": (legacy_freq["best_count"], legacy_freq["total"]),
                "current_freq": (cur_freq["best_count"], cur_freq["total"]),
                "old_freq": (old_freq["best_count"], old_freq["total"]),
            }
        )

        # Show final detections (if available)
        if legacy_detected is not None:
            frame_l, corners_l = legacy_detected
            debug_l = draw_debug(frame_l, corners_l, [])
            cv2.imshow("old_legacy result", debug_l)

        if old_detected is not None:
            frame_o, corners_o = old_detected
            debug_o = draw_debug(frame_o, corners_o, [])
            cv2.imshow("old_better result", debug_o)

        if cur_detected is not None:
            frame_c, res_c = cur_detected
            debug_c = draw_debug(frame_c, res_c.corners, res_c.intersections)
            warped_c = warp_board(frame_c, res_c.corners)
            cv2.imshow("current result", debug_c)
            cv2.imshow("current warped board", warped_c)

        if legacy_rep is not None:
            frame_lr, corners_lr = legacy_rep[1], legacy_rep[2]
            legacy_mode = draw_debug(frame_lr, corners_lr, [])
            cv2.putText(legacy_mode, f"OLD_LEGACY mode: {legacy_freq['best_count']}/{legacy_freq['total']}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow("old_legacy mode result", legacy_mode)

        if old_rep is not None:
            frame_or, corners_or = old_rep[1], old_rep[2]
            old_mode = draw_debug(frame_or, corners_or, [])
            cv2.putText(old_mode, f"OLD mode: {old_freq['best_count']}/{old_freq['total']}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow("old mode result", old_mode)

        if cur_rep is not None:
            frame_cr, corners_cr = cur_rep[1], cur_rep[2]
            cur_mode = draw_debug(frame_cr, corners_cr, [])
            cv2.putText(cur_mode, f"CURRENT mode: {cur_freq['best_count']}/{cur_freq['total']}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow("current mode result", cur_mode)

        if legacy_detected is not None or old_detected is not None or cur_detected is not None:
            print("  -> Result shown. Press any key for next video (ESC to quit all).")
            key2 = cv2.waitKey(0) & 0xFF
            for name in ("old_legacy result", "old_better result", "current result", "current warped board", "old_legacy mode result", "old mode result", "current mode result"):
                try:
                    cv2.destroyWindow(name)
                except Exception:
                    pass
            if key2 == 27:
                break
        else:
            print("  -> No detection in allotted time")

    if overall_summary:
        print("\n=== Overall outcome summary ===")
        for item in overall_summary:
            print(
                f"- {item['video']}: "
                f"old_legacy(metric={item['legacy_metric']:.3f}, hit={item['legacy_hit']:.2f}, freq={item['legacy_freq'][0]}/{item['legacy_freq'][1]}) | "
                f"current(metric={item['current_metric']:.3f}, hit={item['current_hit']:.2f}, freq={item['current_freq'][0]}/{item['current_freq'][1]}) | "
                f"old(metric={item['old_metric']:.3f}, hit={item['old_hit']:.2f}, freq={item['old_freq'][0]}/{item['old_freq'][1]})"
            )

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
