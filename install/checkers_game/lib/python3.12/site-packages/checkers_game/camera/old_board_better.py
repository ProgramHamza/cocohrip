from email.mime import image

import cv2
import numpy as np
from pathlib import Path
import argparse


class OldBoardBetterDetector:
    def piece_recognition(self, warped: np.ndarray) -> tuple[list[tuple[int, int]], dict]:
        """
        Detect occupied squares using edge/gradient/variance features and adaptive splitting.
        Returns: (occupied_squares, square_stats)
        """
        if warped is None or warped.size == 0:
            return [], {}
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        cell_h = h // 8
        cell_w = w // 8
        margin = max(6, min(cell_h, cell_w) // 10)
        stats = []
        occ_scores = []
        square_stats = {}
        for row in range(8):
            for col in range(8):
                if (row + col) % 2 == 0:
                    continue
                y1 = row * cell_h + margin
                y2 = (row + 1) * cell_h - margin
                x1 = col * cell_w + margin
                x2 = (col + 1) * cell_w - margin
                if y2 <= y1 or x2 <= x1:
                    edge_score = 0.0
                    grad_score = 0.0
                    variance = 0.0
                else:
                    roi = gray[y1:y2, x1:x2]
                    edges = cv2.Canny(roi, 50, 150)
                    edge_score = float(np.mean(edges > 0))
                    sobelx = cv2.Sobel(roi, cv2.CV_64F, 1, 0)
                    sobely = cv2.Sobel(roi, cv2.CV_64F, 0, 1)
                    grad_score = float(np.mean(np.sqrt(sobelx ** 2 + sobely ** 2)))
                    variance = float(np.var(roi))
                occ_score = 0.5 * edge_score + 0.5 * grad_score
                stats.append((row, col, occ_score, variance, edge_score, grad_score))
                occ_scores.append(occ_score)
                square_stats[(row, col)] = {
                    "occ_score": occ_score,
                    "variance": variance,
                    "edge_score": edge_score,
                    "grad_score": grad_score,
                }
        occ_scores_np = np.array(occ_scores, dtype=np.float32)
        # Adaptive split: 2 clusters (occupied/unoccupied)
        try:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.2)
            _compactness, labels, centers = cv2.kmeans(
                occ_scores_np.reshape(-1, 1), 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS
            )
            occ_label = int(np.argmax(centers))  # higher score = occupied
            occupied = [i for i, l in enumerate(labels.flatten()) if l == occ_label]
        except Exception:
            # fallback: Otsu or median
            try:
                import skimage.filters
                thresh = skimage.filters.threshold_otsu(occ_scores_np)
            except Exception:
                thresh = float(np.median(occ_scores_np))
            occupied = [i for i, s in enumerate(occ_scores_np) if s > thresh]
        occupied_squares = []
        playable = [(row, col) for row in range(8) for col in range(8) if (row + col) % 2 == 1]
        for idx in occupied:
            occupied_squares.append(playable[idx])
        return occupied_squares, square_stats

    def define_bw_pieces(self, occupied_squares: list[tuple[int, int]], square_stats: dict) -> tuple[np.ndarray, dict]:
        """
        Assign black/white to occupied squares using variance clustering.
        Returns: (board, debug_info)
        """
        board = np.zeros((8, 8), dtype=np.int8)
        if not occupied_squares:
            return board, {"black_count": 0, "white_count": 0, "centers": [], "labels": []}
        variances = np.array([square_stats[sq]["variance"] for sq in occupied_squares], dtype=np.float32)
        try:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.2)
            _compactness, labels, centers = cv2.kmeans(
                variances.reshape(-1, 1), 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS
            )
            # lower variance = black, higher = white
            black_label = int(np.argmin(centers))
            white_label = int(np.argmax(centers))
        except Exception:
            median = float(np.median(variances))
            labels = np.array([0 if v < median else 1 for v in variances])
            black_label, white_label = 0, 1
            centers = np.array([np.mean(variances[labels==0]), np.mean(variances[labels==1])])
        black_count = 0
        white_count = 0
        for idx, sq in enumerate(occupied_squares):
            if labels[idx] == black_label:
                board[sq] = 2
                black_count += 1
            else:
                board[sq] = 1
                white_count += 1
        debug = {
            "black_count": black_count,
            "white_count": white_count,
            "centers": centers.flatten().tolist(),
            "labels": labels.flatten().tolist(),
        }
        return board, debug

    def move_extraction(self, previous_board: np.ndarray, board_now: np.ndarray) -> tuple[int | None, int | None, int | None]:
        """
        Detect a single move (one disappearance, one appearance, same color).
        Returns: (start_square, end_square, color) as flat indices, or (None, None, None) if ambiguous.
        """
        if previous_board is None or board_now is None:
            return None, None, None
        diff = board_now - previous_board
        gone = np.argwhere((previous_board != 0) & (board_now == 0))
        appeared = np.argwhere((previous_board == 0) & (board_now != 0))
        if gone.shape[0] != 1 or appeared.shape[0] != 1:
            return None, None, None
        start = tuple(gone[0])
        end = tuple(appeared[0])
        color_from = previous_board[start]
        color_to = board_now[end]
        if color_from != color_to or color_from not in (1, 2):
            return None, None, None
        return start[0]*8+start[1], end[0]*8+end[1], int(color_from)
  
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        self.parallel_tolerance_deg = 15.0
        self.line_angle_tolerance_deg = 12.0

    def detect_corners(self, image: np.ndarray):
        corners, _debug = self.detect_corners_debug(image)
        return corners

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

    def detect_board_state(self, image: np.ndarray, corners: np.ndarray | None = None):
        if image is None:
            return None, {}

        resolved_corners = corners
        debug = {}

        if resolved_corners is None:
            resolved_corners, corner_debug = self.detect_corners_debug(image)
            debug["corner_debug"] = corner_debug

        if resolved_corners is None:
            return None, debug

        warped = self.warp_board(image, resolved_corners)
        board, board_debug = self.classify_warped_board(warped)
        debug.update(board_debug)
        debug["corners"] = np.array(resolved_corners, dtype=np.float32)
        debug["warped"] = warped
        return board, debug

    def warp_board(self, image: np.ndarray, corners: np.ndarray, board_size: int = 800):
        if image is None or corners is None or len(corners) != 4:
            return None

        dst = np.array(
            [
                [0, 0],
                [board_size - 1, 0],
                [board_size - 1, board_size - 1],
                [0, board_size - 1],
            ],
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
        return cv2.warpPerspective(image, transform, (board_size, board_size))

    def _adaptive_three_class_thresholds(self, values: np.ndarray):
        if values.size == 0:
            return 0.0, 1.0

        flat = values.astype(np.float32).reshape(-1, 1)
        if flat.shape[0] < 3:
            mean_value = float(np.mean(flat))
            return mean_value * 0.8, mean_value * 1.2 + 1.0

        try:
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                25,
                0.2,
            )
            _compactness, _labels, centers = cv2.kmeans(
                flat,
                3,
                None,
                criteria,
                5,
                cv2.KMEANS_PP_CENTERS,
            )
            centers = np.sort(centers.flatten())
            empty_threshold = float((centers[0] + centers[1]) / 2.0)
            black_threshold = float((centers[1] + centers[2]) / 2.0)
        except cv2.error:
            empty_threshold = float(np.percentile(flat, 33))
            black_threshold = float(np.percentile(flat, 66))

        if black_threshold <= empty_threshold:
            spread = max(1.0, float(np.std(flat)))
            black_threshold = empty_threshold + spread

        return empty_threshold, black_threshold

    def classify_warped_board(self, warped: np.ndarray):
        if warped is None or warped.size == 0:
            return None, {}
        # 1. Edge/gradient-based occupancy detection
        occupied_squares, square_stats = self.piece_recognition(warped)
        # 2. Variance-based color split
        board, bw_debug = self.define_bw_pieces(occupied_squares, square_stats)
        # 3. Build debug overlay
        overlay = warped.copy()
        h, w = overlay.shape[:2]
        cell_h = h // 8
        cell_w = w // 8
        for row in range(8):
            for col in range(8):
                x1 = col * cell_w
                y1 = row * cell_h
                x2 = x1 + cell_w
                y2 = y1 + cell_h
                if (row + col) % 2 == 0:
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (70, 70, 70), 1)
                    continue
                value = int(board[row, col])
                if value == 0:
                    color = (0, 255, 0)
                elif value == 2:
                    color = (255, 0, 255)
                else:
                    color = (0, 255, 255)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                variance = square_stats.get((row, col), {}).get("variance", 0.0)
                cv2.putText(
                    overlay,
                    f"{int(variance)}",
                    (x1 + 6, y2 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    color,
                    1,
                )
        cv2.rectangle(overlay, (10, 10), (500, 68), (0, 0, 0), -1)
        cv2.putText(
            overlay,
            f"OldBoardBetter | Black: {bw_debug['black_count']} White: {bw_debug['white_count']}",
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            overlay,
            f"Variance centers: {bw_debug['centers']}",
            (18, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )
        debug = {
            "overlay": overlay,
            "black_count": bw_debug["black_count"],
            "white_count": bw_debug["white_count"],
            "centers": bw_debug["centers"],
            "labels": bw_debug["labels"],
            "square_stats": square_stats,
        }
        return board, debug

    # def piece_recognition(self, image, detect_corners_debug(self, image: np.ndarray).debug['clahe'],warped: np.ndarray):
    # # detect the positions of the pieces
    #     blurred_image = cv2.GaussianBlur(image, (5, 5), 1.4)

    #     square = warped: np.ndarray
    #     edges = cv2.Canny(square, 50, 150)
    #     edge_score = np.sum(edges > 0)
    
    #     sobelx = cv2.Sobel(square, cv2.CV_64F, 1, 0)
    #     sobely = cv2.Sobel(square, cv2.CV_64F, 0, 1)
    #     grad_score = np.mean(np.sqrt(sobelx**2 + sobely**2))
    
    #     occupied_squares = []
    #     for i in square: if (edge_score > t1) or (grad_score > t2): occupied_squares.append(i)

    #     return occupied_squares

    # def define_bw_pieces(self,occupied_squares):
    #     #



    # def move_extraction(self, previous_board, board_now):
        

    #     return start_square, end_square, color

    
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
