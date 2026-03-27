import cv2
import numpy as np
from pathlib import Path
import argparse


class OldBoardBetterDetector:
  
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.parallel_tolerance_deg = 15.0
        self.line_angle_tolerance_deg = 12.0

    def detect_corners_debug(self, image: np.ndarray):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = self.clahe.apply(gray)
        blur = cv2.GaussianBlur(clahe, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        closed = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=2,
        )
        closed_correct = self.lighting_optimization(closed)

        lines = self._detect_lines(closed_correct)
        line_mask = np.zeros_like(closed)
        for x1, y1, x2, y2 in lines:
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)

        # Connect line segments to build closed contour candidates
        line_mask = cv2.dilate(line_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
        line_mask = cv2.morphologyEx(
            line_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
            iterations=1,
        )

        corners = self._largest_quadrilateral(line_mask)
        contour_vis = cv2.cvtColor(line_mask, cv2.COLOR_GRAY2BGR)
        if corners is not None:
            pts = corners.astype(int)
            for i in range(4):
                cv2.line(contour_vis, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 0), 2)

        debug = {
            "gray": gray,
            "clahe": clahe,
            "edges": edges,
            "closed": closed,
            "closed_correct": closed_correct,
            "lines": lines,
            "line_mask": line_mask,
            "contour_vis": contour_vis,
            "corners": corners,
        }
        return corners, debug

    def lighting_optimization(self, closed: np.ndarray) -> np.ndarray:
        contrast_boost = self.clahe.apply(closed)
        _, bw = cv2.threshold(
            contrast_boost,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
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
                # fallback through convex hull approximation
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
        top_candidates = candidates[:4]

        for _area, quad in top_candidates:
            ordered_quad = self._order_points(quad)
            if self._opposite_sides_parallel(ordered_quad, self.parallel_tolerance_deg):
                return ordered_quad

        return None

    def _opposite_sides_parallel(self, quad: np.ndarray, tolerance_deg: float) -> bool:
        def direction_deg(p1: np.ndarray, p2: np.ndarray) -> float:
            v = p2 - p1
            return float(np.degrees(np.arctan2(v[1], v[0])))

        def angle_diff_deg(a: float, b: float) -> float:
            diff = abs(a - b) % 180.0
            return min(diff, 180.0 - diff)

        d01 = direction_deg(quad[0], quad[1])
        d23 = direction_deg(quad[2], quad[3])
        d12 = direction_deg(quad[1], quad[2])
        d30 = direction_deg(quad[3], quad[0])

        pair_1_parallel = angle_diff_deg(d01, d23) <= tolerance_deg
        pair_2_parallel = angle_diff_deg(d12, d30) <= tolerance_deg
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


class TemporalBoardStabilizer:
    def __init__(self, hold_frames: int = 3):
        self.hold_frames = max(0, int(hold_frames))
        self.last_corners = None
        self.missed_frames = 0

    def update(self, corners: np.ndarray | None):
        if corners is not None:
            self.last_corners = np.array(corners, dtype=np.float32).reshape(4, 2)
            self.missed_frames = 0
            return self.last_corners, False

        if self.last_corners is not None and self.missed_frames < self.hold_frames:
            self.missed_frames += 1
            return self.last_corners.copy(), True

        self.last_corners = None
        self.missed_frames = 0
        return None, False


list_of_videos = [
    "2026-02-25_17-27-27.mp4",
    "2026-02-25_17-29-20.mp4",
    "2026-02-25_17-32-05.mp4",
    # "fill names of the videos here.mp4",
]


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


def test(video_idx: int, slowmo: int, hold_frames: int = 3):
    if not list_of_videos:
        raise ValueError("list_of_videos is empty. Add video names first.")

    if video_idx < 0 or video_idx >= len(list_of_videos):
        raise IndexError(f"video_idx must be in range 0..{len(list_of_videos) - 1}")

    video_path = _resolve_video_path(list_of_videos[video_idx])
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    detector = OldBoardBetterDetector()
    stabilizer = TemporalBoardStabilizer(hold_frames=hold_frames)
    frame_idx = 0

    def _to_bgr(img: np.ndarray) -> np.ndarray:
        if img is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    cv2.namedWindow("old_board_better_debug", cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        corners, debug = detector.detect_corners_debug(frame)
        stabilized_corners, used_hold = stabilizer.update(corners)

        vis = frame.copy()
        if stabilized_corners is not None:
            pts = stabilized_corners.astype(int)
            for i in range(4):
                cv2.line(vis, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 0), 2)

        h, w = frame.shape[:2]
        stages = [
            vis,
            _to_bgr(debug.get("gray")),
            _to_bgr(debug.get("clahe")),
            _to_bgr(debug.get("edges")),
            _to_bgr(debug.get("closed")),
            _to_bgr(debug.get("closed_correct")),
            _to_bgr(debug.get("line_mask")),
            _to_bgr(debug.get("contour_vis")),
        ]

        stages = [cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA) for img in stages]
        top = np.hstack(stages[:4])
        bottom = np.hstack(stages[4:])
        panel = np.vstack([top, bottom])

        if corners is not None:
            status = "DETECTED"
            color = (0, 255, 0)
        elif used_hold:
            status = "STABLE_HOLD"
            color = (0, 255, 255)
        else:
            status = "SEARCHING"
            color = (0, 165, 255)

        cv2.putText(panel, f"video_idx={video_idx} frame={frame_idx} status={status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(panel, f"Views: vis(stabilized) | gray | clahe | edges | closed | closed_correct | line_mask | contour_vis", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(panel, f"Temporal hold frames={hold_frames}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("old_board_better_debug", panel)
        key = cv2.waitKey(max(1, int(slowmo))) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


def _parse_args():
    parser = argparse.ArgumentParser(description="Run OldBoardBetter detector debug on prerecorded videos")
    parser.add_argument("--video-idx", type=int, default=0, help="Index in list_of_videos")
    parser.add_argument("--slowmo", type=int, default=120, help="Delay per frame in ms")
    parser.add_argument("--hold-frames", type=int, default=3, help="How many missed frames to keep last valid board")
    parser.add_argument("--list-videos", action="store_true", help="Print list_of_videos and exit")
    return parser.parse_args()


def main():
    args = _parse_args()

    if args.list_videos:
        for idx, name in enumerate(list_of_videos):
            print(f"[{idx}] {name}")
        return

    test(video_idx=args.video_idx, slowmo=args.slowmo, hold_frames=args.hold_frames)


if __name__ == "__main__":
    main()
