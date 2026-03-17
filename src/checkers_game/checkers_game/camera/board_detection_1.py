import argparse
from collections import Counter
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from .old_board_better import OldBoardBetterDetector, TemporalBoardStabilizer

try:
    from .ximea_camera import XimeaCamera
except Exception:
    XimeaCamera = None


list_of_videos = [
    "2026-02-25_17-27-27.mp4",
    "2026-02-25_17-29-20.mp4",
    "2026-02-25_17-32-05.mp4",
]


class BoardDetector:
    def __init__(self):
        self.reference_detector = OldBoardBetterDetector()

    def detect_corners_debug(self, image: np.ndarray):
        return self.reference_detector.detect_corners_debug(image)

    def lighting_optimization(self, closed: np.ndarray) -> np.ndarray:
        contrast_boost = self.clahe.apply(closed)
        _, bw = cv2.threshold(contrast_boost, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bw = cv2.morphologyEx(
            bw,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )
        return bw

    def _detect_lines(self, edge_image: np.ndarray):
        h, w = edge_image.shape[:2]
        min_len = int(0.18 * max(h, w))
        lines = cv2.HoughLinesP(
            edge_image,
            rho=1,
            theta=np.pi / 180,
            threshold=110,
            minLineLength=min_len,
            maxLineGap=12,
        )
        if lines is None:
            return []

        raw_lines = [tuple(map(int, l[0])) for l in lines]
        length_filtered = [line for line in raw_lines if self._line_length(line) >= min_len]
        if not length_filtered:
            return []

        filtered = self._filter_lines_by_dominant_orientations(length_filtered)
        filtered.sort(key=self._line_length, reverse=True)
        return filtered[:80]

    def _line_length(self, line) -> float:
        x1, y1, x2, y2 = line
        return float(np.hypot(x2 - x1, y2 - y1))

    def _line_angle_deg(self, line) -> float:
        x1, y1, x2, y2 = line
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        return angle % 180.0

    def _angle_diff_180(self, a: float, b: float) -> float:
        diff = abs(a - b) % 180.0
        return min(diff, 180.0 - diff)

    def _filter_lines_by_dominant_orientations(self, lines):
        if len(lines) <= 2:
            return lines

        bin_size = 10.0
        num_bins = int(180 / bin_size)
        weighted_bins = np.zeros(num_bins, dtype=np.float32)

        for line in lines:
            angle = self._line_angle_deg(line)
            idx = int(angle // bin_size) % num_bins
            weighted_bins[idx] += self._line_length(line)

        primary_idx = int(np.argmax(weighted_bins))
        primary_angle = (primary_idx + 0.5) * bin_size

        secondary_idx = None
        secondary_score = -1.0
        for idx, score in enumerate(weighted_bins):
            candidate_angle = (idx + 0.5) * bin_size
            if self._angle_diff_180(candidate_angle, primary_angle) < 25.0:
                continue
            if score > secondary_score:
                secondary_score = float(score)
                secondary_idx = idx

        keep_angles = [primary_angle]
        if secondary_idx is not None and secondary_score > 0.0:
            keep_angles.append((secondary_idx + 0.5) * bin_size)

        filtered = []
        for line in lines:
            angle = self._line_angle_deg(line)
            if any(self._angle_diff_180(angle, target) <= self.line_angle_tolerance_deg for target in keep_angles):
                filtered.append(line)

        return filtered if filtered else lines

    def _largest_quadrilateral(self, binary_image: np.ndarray):
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 12000:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) != 4:
                hull = cv2.convexHull(cnt)
                peri_h = cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, 0.02 * peri_h, True)
                if len(approx) != 4:
                    continue

            quad = approx.reshape(4, 2).astype(np.float32)
            candidates.append((area, quad))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        for _area, quad in candidates[:10]:
            ordered_quad = self._order_points(quad)
            if self._opposite_sides_parallel(ordered_quad, self.parallel_tolerance_deg):
                return ordered_quad
        return None

    def _opposite_sides_parallel(self, quad: np.ndarray, tolerance_deg: float) -> bool:
        def direction_deg(p1: np.ndarray, p2: np.ndarray) -> float:
            v = p2 - p1
            return float(np.degrees(np.arctan2(v[1], v[0])))

        d01 = direction_deg(quad[0], quad[1])
        d23 = direction_deg(quad[2], quad[3])
        d12 = direction_deg(quad[1], quad[2])
        d30 = direction_deg(quad[3], quad[0])

        pair_1_parallel = self._angle_diff_180(d01, d23) <= tolerance_deg
        pair_2_parallel = self._angle_diff_180(d12, d30) <= tolerance_deg
        return pair_1_parallel and pair_2_parallel

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect


def _resolve_video_path(video_name: str) -> Path:
    video_path = Path(video_name).expanduser()
    if video_path.exists():
        return video_path

    if not video_path.is_absolute():
        candidate = Path.cwd() / video_path
        if candidate.exists():
            return candidate

    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        candidate = parent / "videos_training" / video_path.name
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Video not found: {video_name}")


def _warp_board(frame: np.ndarray, corners: np.ndarray, size: int = 800) -> np.ndarray:
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(frame, matrix, (size, size))


def _classify_board_cells(warped: np.ndarray):
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    cell_size = warped.shape[0] // 8
    margin = max(6, cell_size // 10)

    variances = []
    for row in range(8):
        for col in range(8):
            y1 = row * cell_size + margin
            y2 = (row + 1) * cell_size - margin
            x1 = col * cell_size + margin
            x2 = (col + 1) * cell_size - margin
            roi = gray[y1:y2, x1:x2]
            variances.append(float(np.var(roi)) if roi.size > 0 else 0.0)

    all_vars = np.array(variances, dtype=np.float32)
    empty_threshold = float(np.percentile(all_vars, 40))
    black_threshold = float(np.percentile(all_vars, 75))

    labels = []
    idx = 0
    for _row in range(8):
        row_labels = []
        for _col in range(8):
            var = variances[idx]
            idx += 1
            if var < empty_threshold:
                row_labels.append(("E", var))
            elif var < black_threshold:
                row_labels.append(("B", var))
            else:
                row_labels.append(("W", var))
        labels.append(row_labels)

    return labels, empty_threshold, black_threshold


def _draw_board_overlay(warped: np.ndarray, labels) -> np.ndarray:
    vis = warped.copy()
    cell_size = warped.shape[0] // 8

    for row in range(8):
        for col in range(8):
            x1 = col * cell_size
            y1 = row * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size

            label, variance = labels[row][col]
            if label == "E":
                color = (0, 255, 0)
            elif label == "B":
                color = (255, 0, 255)
            else:
                color = (255, 255, 0)

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, label, (x1 + 8, y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(vis, f"{int(variance)}", (x1 + 8, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return vis


def _consensus_corners(corner_candidates, bin_size: float = 8.0) -> Optional[np.ndarray]:
    if not corner_candidates:
        return None

    consensus = np.zeros((4, 2), dtype=np.float32)
    for corner_idx in range(4):
        point_bins = []
        point_map = {}

        for corners in corner_candidates:
            point = corners[corner_idx]
            point_key = (
                int(round(float(point[0]) / bin_size)),
                int(round(float(point[1]) / bin_size)),
            )
            point_bins.append(point_key)
            if point_key not in point_map:
                point_map[point_key] = np.array(point, dtype=np.float32)

        winning_key, _winning_votes = Counter(point_bins).most_common(1)[0]
        consensus[corner_idx] = point_map[winning_key]

    return consensus


def _render_pipeline_panel(frame: np.ndarray, fixed_corners: Optional[np.ndarray]):
    frame_vis = frame.copy()
    board_vis = np.zeros((800, 800, 3), dtype=np.uint8)

    if fixed_corners is not None:
        pts = fixed_corners.astype(int)
        for i in range(4):
            cv2.line(frame_vis, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 0), 2)

        warped = _warp_board(frame, fixed_corners, size=800)
        labels, empty_thr, black_thr = _classify_board_cells(warped)
        board_vis = _draw_board_overlay(warped, labels)
        cv2.putText(board_vis, f"E<{empty_thr:.1f} <B<{black_thr:.1f} <W", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    left = cv2.resize(frame_vis, (800, 800), interpolation=cv2.INTER_AREA)
    panel = np.hstack([left, board_vis])
    return panel


def run_on_video(
    video_idx: int,
    slowmo: int = 120,
    consensus_frames: int = 20,
    lock_mode: str = "raw",
    hold_frames: int = 3,
):
    if video_idx < 0 or video_idx >= len(list_of_videos):
        raise IndexError(f"video_idx must be in range 0..{len(list_of_videos) - 1}")

    video_path = _resolve_video_path(list_of_videos[video_idx])
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    detector = BoardDetector()
    stabilizer = TemporalBoardStabilizer(hold_frames=hold_frames)
    fixed_corners = None
    frame_count_for_consensus = 0
    corner_candidates = []
    cv2.namedWindow("board_detection_1", cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if lock_mode == "raw":
            corners, _debug = detector.detect_corners_debug(frame)
            preview_corners, _used_hold = stabilizer.update(corners)
            panel = _render_pipeline_panel(frame, preview_corners)
            status = "DETECTED" if corners is not None else ("HOLD" if preview_corners is not None else "SEARCHING")
            color = (0, 255, 0) if status == "DETECTED" else ((0, 255, 255) if status == "HOLD" else (0, 165, 255))
            cv2.putText(panel, f"Grid points: {status} (raw mode)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(panel, f"Hold frames: {hold_frames}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        else:
            preview_corners = fixed_corners
            if fixed_corners is None and frame_count_for_consensus < consensus_frames:
                corners, _debug = detector.detect_corners_debug(frame)
                frame_count_for_consensus += 1
                if corners is not None:
                    corner_candidates.append(corners.copy())
                    preview_corners = corners.copy()

                if frame_count_for_consensus >= consensus_frames:
                    fixed_corners = _consensus_corners(corner_candidates)
                    preview_corners = fixed_corners
                    if fixed_corners is None:
                        frame_count_for_consensus = 0
                        corner_candidates.clear()

            panel = _render_pipeline_panel(frame, preview_corners)
            if fixed_corners is not None:
                lock_status = "LOCKED"
                lock_color = (0, 255, 0)
            elif frame_count_for_consensus < consensus_frames:
                lock_status = f"VOTING {frame_count_for_consensus}/{consensus_frames}"
                lock_color = (0, 165, 255)
            else:
                lock_status = "SEARCHING"
                lock_color = (0, 165, 255)
            cv2.putText(panel, f"Grid points: {lock_status} (vote mode)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, lock_color, 2)
            cv2.putText(panel, f"Vote hits: {len(corner_candidates)}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(panel, "Left: source+corners | Right: warped board + 8x8 piece classification", (20, 770), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(panel, "Keys: q/ESC=quit, r=reset grid lock", (20, 795), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imshow("board_detection_1", panel)

        key = cv2.waitKey(max(1, int(slowmo))) & 0xFF
        if key in (ord("r"), ord("R")):
            fixed_corners = None
            frame_count_for_consensus = 0
            corner_candidates.clear()
            stabilizer = TemporalBoardStabilizer(hold_frames=hold_frames)
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_on_ximea(
    slowmo: int = 1,
    consensus_frames: int = 20,
    lock_mode: str = "raw",
    hold_frames: int = 3,
):
    if XimeaCamera is None:
        raise RuntimeError("XimeaCamera is not available. Ensure ximea SDK/python package is installed.")

    camera = XimeaCamera()
    detector = BoardDetector()
    stabilizer = TemporalBoardStabilizer(hold_frames=hold_frames)
    fixed_corners = None
    frame_count_for_consensus = 0
    corner_candidates = []
    cv2.namedWindow("board_detection_1", cv2.WINDOW_NORMAL)

    while True:
        frame = camera.get_camera_image()
        if frame is None:
            continue

        if lock_mode == "raw":
            corners, _debug = detector.detect_corners_debug(frame)
            preview_corners, _used_hold = stabilizer.update(corners)
            panel = _render_pipeline_panel(frame, preview_corners)
            status = "DETECTED" if corners is not None else ("HOLD" if preview_corners is not None else "SEARCHING")
            color = (0, 255, 0) if status == "DETECTED" else ((0, 255, 255) if status == "HOLD" else (0, 165, 255))
            cv2.putText(panel, f"Grid points: {status} (raw mode)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(panel, f"Hold frames: {hold_frames}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        else:
            preview_corners = fixed_corners
            if fixed_corners is None and frame_count_for_consensus < consensus_frames:
                corners, _debug = detector.detect_corners_debug(frame)
                frame_count_for_consensus += 1
                if corners is not None:
                    corner_candidates.append(corners.copy())
                    preview_corners = corners.copy()

                if frame_count_for_consensus >= consensus_frames:
                    fixed_corners = _consensus_corners(corner_candidates)
                    preview_corners = fixed_corners
                    if fixed_corners is None:
                        frame_count_for_consensus = 0
                        corner_candidates.clear()

            panel = _render_pipeline_panel(frame, preview_corners)
            if fixed_corners is not None:
                lock_status = "LOCKED"
                lock_color = (0, 255, 0)
            elif frame_count_for_consensus < consensus_frames:
                lock_status = f"VOTING {frame_count_for_consensus}/{consensus_frames}"
                lock_color = (0, 165, 255)
            else:
                lock_status = "SEARCHING"
                lock_color = (0, 165, 255)
            cv2.putText(panel, f"Grid points: {lock_status} (vote mode)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, lock_color, 2)
            cv2.putText(panel, f"Vote hits: {len(corner_candidates)}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.putText(panel, "Left: source+corners | Right: warped board + 8x8 piece classification", (20, 770), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(panel, "Keys: q/ESC=quit, r=reset grid lock", (20, 795), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imshow("board_detection_1", panel)

        key = cv2.waitKey(max(1, int(slowmo))) & 0xFF
        if key in (ord("r"), ord("R")):
            fixed_corners = None
            frame_count_for_consensus = 0
            corner_candidates.clear()
            stabilizer = TemporalBoardStabilizer(hold_frames=hold_frames)
        if key in (27, ord("q")):
            break

    cv2.destroyAllWindows()


def _parse_args():
    parser = argparse.ArgumentParser(description="board_detection_1: Ximea/video corner + grid + piece-color demo")
    parser.add_argument("--mode", type=str, default="video", choices=["video", "ximea"], help="Input source")
    parser.add_argument("--video-idx", type=int, default=0, help="Index in list_of_videos for --mode video")
    parser.add_argument("--slowmo", type=int, default=120, help="Frame delay in ms")
    parser.add_argument("--lock-mode", type=str, default="raw", choices=["raw", "vote"], help="raw matches old_board_better behavior, vote locks corners from voting window")
    parser.add_argument("--hold-frames", type=int, default=3, help="Used in raw mode to hold last valid detection")
    parser.add_argument("--consensus-frames", type=int, default=20, help="How many first frames are used for corner voting")
    parser.add_argument("--list-videos", action="store_true", help="Print available videos and exit")
    return parser.parse_args()


def main():
    args = _parse_args()

    if args.list_videos:
        for idx, name in enumerate(list_of_videos):
            print(f"[{idx}] {name}")
        return

    if args.mode == "ximea":
        run_on_ximea(
            slowmo=args.slowmo,
            consensus_frames=max(1, int(args.consensus_frames)),
            lock_mode=args.lock_mode,
            hold_frames=max(0, int(args.hold_frames)),
        )
        return

    run_on_video(
        video_idx=args.video_idx,
        slowmo=args.slowmo,
        consensus_frames=max(1, int(args.consensus_frames)),
        lock_mode=args.lock_mode,
        hold_frames=max(0, int(args.hold_frames)),
    )


if __name__ == "__main__":
    main()
