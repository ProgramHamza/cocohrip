import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .grid_corner_detector import GridCornerDetector
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
    """Board detector compatible with board_detection.py flow, using OldBoardBetter runtime recognition."""

    def __init__(self, ximeaCamera=None, hold_frames: int = 3):
        self.ximeaCamera = ximeaCamera
        self.grid_detector = GridCornerDetector()
        self.old_board_detector = OldBoardBetterDetector()
        self.temporal_stabilizer = TemporalBoardStabilizer(hold_frames=max(0, int(hold_frames)))

        # Keep behavior and attributes expected by checkers_node/game.
        self.use_old_board_better_runtime = True
        self._old_board_runtime_reported = False
        self.selected_difficulty = 3
        self.numberOfEmptyFields = 40
        self.param1ForGetAllContours = 255
        self.gameBoardFieldsContours = self._get_grid_squares_contours()
        self.is_initialized = False

        self.empty_variance_threshold = 15.0
        self.black_variance_threshold = 1000.0
        self.white_piece_threshold = 1000.0

        self.bounderies = None
        self.expected_black_left = 12
        self.expected_white_left = 12
        self._last_board = None
        self._last_selected_method = "uninitialized"

        self._load_threshold_profile_if_available()

        if self.ximeaCamera is not None:
            self._init()

    def _load_threshold_profile_if_available(self):
        profile_path = Path(__file__).resolve().parent / "evaluation_output" / "copy_threshold_calibration.json"
        if not profile_path.exists():
            return
        try:
            with profile_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.empty_variance_threshold = float(payload.get("empty_variance_threshold", self.empty_variance_threshold))
            self.black_variance_threshold = float(payload.get("black_variance_threshold", self.black_variance_threshold))
            self.white_piece_threshold = float(payload.get("white_piece_threshold", self.white_piece_threshold))
            print(
                "Loaded threshold profile into bd1: "
                f"E<{self.empty_variance_threshold:.2f}<B<{self.black_variance_threshold:.2f}<W({self.white_piece_threshold:.2f})"
            )
        except Exception as exc:
            print(f"Failed to load threshold profile for bd1: {exc}")

    def _init(self):
        # 1. Camera Adjustment Phase
        self._camera_adjustment_window()

        # 2. Board Corner Detection (AUTO)
        print("\nAttempting automatic board detection...")
        auto_corners = self._auto_detect_corners()

        if auto_corners is not None:
            self.bounderies = auto_corners
            print("? Automatic detection successful!")
        else:
            print("? Automatic detection failed. Falling back to manual selection.")
            self.bounderies = self._get_trim_param_manual()

        # 3. Piece placement verification window
        self._piece_placement_window()

        # 4. Final initialization
        self.numberOfEmptyFields = 40
        self.param1ForGetAllContours = 255
        self.gameBoardFieldsContours = self._get_grid_squares_contours()

        self.is_initialized = False
        self.selected_difficulty = 3
        print("Default runtime board detector: OldBoardBetter")

    def detect_corners_debug(self, image: np.ndarray):
        return self.old_board_detector.detect_corners_debug(image)

    def _auto_detect_corners(self):
        if self.ximeaCamera is None:
            return None
        image = self.ximeaCamera.get_camera_image()
        return self._auto_detect_corners_from_image(image)

    def _auto_detect_corners_from_image(self, image):
        if image is None:
            return None

        # 1. Try grid-based Hough detection
        grid_result = self.grid_detector.detect_corners(image)
        if grid_result is not None:
            print("  → Grid-based detection succeeded")
            return self._orient_corners(grid_result.corners, image)

        # 2. Try OldBoardBetter low-light robust detector
        old_corners, _old_debug = self.old_board_detector.detect_corners_debug(image)
        if old_corners is not None:
            print("  → OldBoardBetter detection succeeded")
            return self._orient_corners(old_corners, image)

        # 3. Contour-based fallback
        fallback = self._auto_detect_corners_contour_fallback(image)
        if fallback is not None:
            print("  → Contour fallback detection succeeded")
            return fallback

        return None

    def _auto_detect_corners_contour_fallback(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2,
        )

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largest_area = 0
        board_cnt = None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50000:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4 and area > largest_area:
                largest_area = area
                board_cnt = approx

        if board_cnt is None:
            return None

        pts = board_cnt.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return self._orient_corners(rect, image)

    def _orient_corners(self, corners, image):
        """Rotate corner order to best match expected board setup orientation."""
        best_corners = corners
        best_score = float("inf")

        for _ in range(4):
            warped = self._trim_image_perspective(image, corners)
            if warped is None or warped.size == 0:
                break

            roi_tl = warped[10:90, 10:90]
            roi_tr = warped[10:90, 710:790]
            roi_br = warped[710:790, 710:790]
            roi_bl = warped[710:790, 10:90]

            v_tl = self._calculate_variance(roi_tl)
            v_tr = self._calculate_variance(roi_tr)
            v_br = self._calculate_variance(roi_br)
            v_bl = self._calculate_variance(roi_bl)

            score = 0.0
            score += v_tl + v_br
            if v_tr < 200:
                score += 10000
            if v_bl < 200:
                score += 10000
            if v_bl < v_tr:
                score += 5000

            if score < best_score:
                best_score = score
                best_corners = corners.copy()

            corners = np.roll(corners, 1, axis=0)

        print(f"  -> Oriented corners with score: {best_score}")
        return best_corners

    def _camera_adjustment_window(self):
        if self.ximeaCamera is None:
            return

        print("\n" + "=" * 60)
        print("STEP 0: CAMERA ADJUSTMENT")
        print("=" * 60)
        print("Adjust the camera so the whole board is visible.")
        print("Press SPACE to continue when ready.")
        print("-" * 60 + "\n")

        while True:
            image = self.ximeaCamera.get_camera_image()
            if image is None:
                continue

            display = image.copy()
            cv2.putText(
                display,
                "Adjust Camera. Press SPACE to continue",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Camera Adjustment", display)

            if cv2.waitKey(1) & 0xFF == 32:
                cv2.destroyWindow("Camera Adjustment")
                break

    def _calculate_variance(self, image):
        if image is None or image.size == 0:
            return 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _mean, stddev = cv2.meanStdDev(gray)
        return float(stddev[0][0] ** 2)

    def _piece_placement_window(self):
        if self.ximeaCamera is None:
            return

        print("\n" + "=" * 60)
        print("STEP 3: PIECE PLACEMENT CHECK")
        print("=" * 60)
        print("Place pieces in starting positions.")
        print("Press SPACE or ENTER to confirm and Start Game.")
        print("-" * 60 + "\n")

        while True:
            image = self.ximeaCamera.get_camera_image()
            if image is None:
                continue
            if self.bounderies is None:
                continue

            warped = self._trim_image_perspective(image, self.bounderies)
            if warped is None:
                continue

            board, overlay, black_count, white_count, empty_thr, black_thr = self._classify_warped_board(warped)
            if board is None:
                display = warped.copy()
                cv2.putText(display, "Recognition failed", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                cv2.imshow("Piece Placement & Detection", display)
            else:
                display = overlay.copy()
                cv2.putText(
                    display,
                    f"Black: {black_count} | White: {white_count}",
                    (20, 770),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    display,
                    f"Thresholds E<{empty_thr:.1f}<B<{black_thr:.1f}<W",
                    (20, 795),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                )
                cv2.imshow("Piece Placement & Detection", display)

            key = cv2.waitKey(10) & 0xFF
            if key in (13, 32):
                cv2.destroyWindow("Piece Placement & Detection")
                print("? Board setup confirmed! Starting game...\n")
                break
            if key == 27:
                cv2.destroyWindow("Piece Placement & Detection")
                print("? Detection skipped by user.")
                break

    def _extract_dark_square_variances(self, warped: np.ndarray):
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        h, w = blur.shape[:2]
        cell_h = h // 8
        cell_w = w // 8
        margin = max(6, min(cell_h, cell_w) // 10)

        stats = []
        for row in range(8):
            for col in range(8):
                y1 = row * cell_h + margin
                y2 = (row + 1) * cell_h - margin
                x1 = col * cell_w + margin
                x2 = (col + 1) * cell_w - margin

                if y2 <= y1 or x2 <= x1:
                    variance = 0.0
                else:
                    roi = blur[y1:y2, x1:x2]
                    variance = float(np.var(roi)) if roi.size > 0 else 0.0

                stats.append((row, col, variance))

        return stats, cell_h, cell_w

    def _estimate_black_threshold_from_occupied(self, occupied_vars: np.ndarray, default_value: float) -> float:
        if occupied_vars.size < 2:
            return float(default_value)
        try:
            vals = occupied_vars.astype(np.float32).reshape(-1, 1)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.2)
            _compactness, _labels, centers = cv2.kmeans(vals, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
            centers = np.sort(centers.flatten())
            return float((centers[0] + centers[1]) / 2.0)
        except Exception:
            return float(np.percentile(occupied_vars, 50))

    def _build_board_from_thresholds(self, warped: np.ndarray, stats, cell_h: int, cell_w: int, empty_thr: float, black_thr: float):
        board = np.zeros((8, 8), dtype=np.int8)
        black_count = 0
        white_count = 0

        for row, col, variance in stats:
            if variance < empty_thr:
                board[row, col] = 0
            elif variance < black_thr:
                board[row, col] = 2
                black_count += 1
            else:
                board[row, col] = 1
                white_count += 1

        overlay = warped.copy()
        position = 0
        for row in range(8):
            for col in range(8):
                x1 = col * cell_w
                y1 = row * cell_h
                x2 = x1 + cell_w
                y2 = y1 + cell_h

                value = int(board[row, col])
                if value == 0:
                    color = (0, 255, 0)
                    label = "E"
                elif value == 2:
                    color = (255, 0, 255)
                    label = "B"
                else:
                    color = (0, 255, 255)
                    label = "W"

                # Dim border for light squares, thicker for dark
                if (row + col) % 2 == 0:
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (70, 70, 70), 1)
                else:
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

                # Draw position number centered in the square
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                pos_text = str(position)
                pos_size = cv2.getTextSize(pos_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                px = cx - pos_size[0] // 2
                py = cy - 5
                cv2.rectangle(overlay,
                             (px - 2, py - pos_size[1] - 2),
                             (px + pos_size[0] + 2, py + 2),
                             (0, 0, 0), -1)
                cv2.putText(overlay, pos_text, (px, py),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Draw label (E/B/W) below position number
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                lx = cx - label_size[0] // 2
                ly = cy + label_size[1] + 10
                cv2.rectangle(overlay,
                             (lx - 3, ly - label_size[1] - 3),
                             (lx + label_size[0] + 3, ly + 3),
                             (0, 0, 0), -1)
                cv2.putText(overlay, label, (lx, ly),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Draw variance value at top of square
                var_val = [v for r, c, v in stats if r == row and c == col]
                if var_val:
                    var_text = f"{int(var_val[0])}"
                    var_size = cv2.getTextSize(var_text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
                    vx = cx - var_size[0] // 2
                    vy = y1 + var_size[1] + 5
                    cv2.rectangle(overlay,
                                 (vx - 2, vy - var_size[1] - 2),
                                 (vx + var_size[0] + 2, vy + 2),
                                 (0, 0, 0), -1)
                    cv2.putText(overlay, var_text, (vx, vy),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

                position += 1

        return board.astype(object), overlay, int(black_count), int(white_count)

    def _score_candidate(self, board: np.ndarray, black_count: int, white_count: int) -> float:
        expected_black = int(max(0, self.expected_black_left))
        expected_white = int(max(0, self.expected_white_left))
        expected_dark_empty = int(max(0, min(32, self.numberOfEmptyFields - 32)))
        dark_empty = 32 - int(black_count) - int(white_count)

        score = 0.0
        score += abs(int(black_count) - expected_black) * 3.0
        score += abs(int(white_count) - expected_white) * 3.0
        score += abs(int(dark_empty) - expected_dark_empty) * 2.0

        if self._last_board is not None and isinstance(self._last_board, np.ndarray):
            changed = float(np.count_nonzero(self._last_board != board))
            # Penalize noisy jumps, but allow genuine move changes.
            if changed > 8:
                score += (changed - 8.0) * 0.5

        return score

    def _classify_warped_board(self, warped: np.ndarray):
        if warped is None:
            return None, None, 0, 0, 0.0, 0.0

        stats, cell_h, cell_w = self._extract_dark_square_variances(warped)
        if len(stats) == 0:
            return None, None, 0, 0, 0.0, 0.0
        var_array = np.array([v for _, _, v in stats], dtype=np.float32)

        candidates = []

        # Candidate 1: OldBoardBetter direct output
        classify_fn = getattr(self.old_board_detector, "classify_warped_board", None)
        if callable(classify_fn):
            try:
                board_old, debug = classify_fn(warped)
                if board_old is not None:
                    board_old = board_old.astype(object)
                    b_old = int(debug.get("black_count", int(np.count_nonzero(board_old == 2))))
                    w_old = int(debug.get("white_count", int(np.count_nonzero(board_old == 1))))
                    e_old = float(debug.get("empty_threshold", self.empty_variance_threshold))
                    bt_old = float(debug.get("black_threshold", self.black_variance_threshold))
                    ov_old = debug.get("overlay", warped.copy())
                    score_old = self._score_candidate(np.array(board_old, dtype=np.int32), b_old, w_old)
                    candidates.append(("old_board_better", score_old, board_old, ov_old, b_old, w_old, e_old, bt_old))
            except Exception:
                pass

        # Candidate 2: fixed/profile thresholds
        e_fixed = float(self.empty_variance_threshold)
        b_fixed = float(max(self.black_variance_threshold, e_fixed + 1.0))
        board_fixed, ov_fixed, b_fixed_n, w_fixed_n = self._build_board_from_thresholds(
            warped, stats, cell_h, cell_w, e_fixed, b_fixed
        )
        score_fixed = self._score_candidate(np.array(board_fixed, dtype=np.int32), b_fixed_n, w_fixed_n)
        candidates.append(("fixed_thresholds", score_fixed, board_fixed, ov_fixed, b_fixed_n, w_fixed_n, e_fixed, b_fixed))

        # Candidate 3: adaptive percentiles
        e_adapt = float(np.percentile(var_array, 33))
        b_adapt = float(np.percentile(var_array, 66))
        if b_adapt <= e_adapt:
            b_adapt = e_adapt + max(1.0, float(np.std(var_array)))
        board_adapt, ov_adapt, b_adapt_n, w_adapt_n = self._build_board_from_thresholds(
            warped, stats, cell_h, cell_w, e_adapt, b_adapt
        )
        score_adapt = self._score_candidate(np.array(board_adapt, dtype=np.int32), b_adapt_n, w_adapt_n)
        candidates.append(("adaptive_percentiles", score_adapt, board_adapt, ov_adapt, b_adapt_n, w_adapt_n, e_adapt, b_adapt))

        # Candidate 4: expected-empty constrained split
        expected_dark_empty = int(max(0, min(32, self.numberOfEmptyFields - 32)))
        sorted_vars = np.sort(var_array)
        if expected_dark_empty <= 0:
            e_expected = float(sorted_vars[0] - 1e-6)
        elif expected_dark_empty >= len(sorted_vars):
            e_expected = float(sorted_vars[-1] + 1e-6)
        else:
            e_expected = float((sorted_vars[expected_dark_empty - 1] + sorted_vars[expected_dark_empty]) / 2.0)

        occupied_vars = var_array[var_array >= e_expected]
        b_expected = self._estimate_black_threshold_from_occupied(
            occupied_vars,
            default_value=max(self.black_variance_threshold, e_expected + 1.0),
        )
        if b_expected <= e_expected:
            b_expected = e_expected + 1.0

        board_expected, ov_expected, b_expected_n, w_expected_n = self._build_board_from_thresholds(
            warped, stats, cell_h, cell_w, e_expected, b_expected
        )
        score_expected = self._score_candidate(np.array(board_expected, dtype=np.int32), b_expected_n, w_expected_n)
        candidates.append(
            ("expected_empty_split", score_expected, board_expected, ov_expected, b_expected_n, w_expected_n, e_expected, b_expected)
        )

        if len(candidates) == 0:
            return None, None, 0, 0, 0.0, 0.0

        selected = min(candidates, key=lambda x: x[1])
        method_name, _score, board, overlay, black_count, white_count, empty_thr, black_thr = selected
        self._last_selected_method = method_name
        self._last_board = np.array(board, dtype=np.int32)
        return board, overlay, black_count, white_count, float(empty_thr), float(black_thr)

    def _get_grid_squares_contours(self):
        contours = []
        cell_size = 100
        for row in range(8):
            row_cnts = []
            for col in range(8):
                x = col * cell_size
                y = row * cell_size
                row_cnts.append([x, y, cell_size, cell_size])
            contours.append(row_cnts)
        return contours

    def get_board(self, cameraImage, game):
        """Main runtime entry point, compatible with board_detection.py."""
        if cameraImage is None:
            return None

        if self.bounderies is None:
            auto_corners = self._auto_detect_corners_from_image(cameraImage)
            if auto_corners is None:
                return None
            self.bounderies = auto_corners

        warped = self._trim_image_perspective(cameraImage, self.bounderies)
        if warped is None:
            return None

        if not hasattr(self, "gameBoardFieldsContours") or self.gameBoardFieldsContours is None:
            self.gameBoardFieldsContours = self._get_grid_squares_contours()

        self.expected_black_left = int(max(0, getattr(game.board, "black_left", 12)))
        self.expected_white_left = int(max(0, getattr(game.board, "white_left", 12)))
        self.set_number_of_empty_fields(game)

        if self.use_old_board_better_runtime:
            board = self._get_board_with_old_board_better(warped)
            if board is not None:
                return board

        # Fallback: legacy threshold board extraction.
        return self._get_board_from_image_fallback(warped)

    def _get_board_with_old_board_better(self, warped_board_image):
        board, overlay, black_count, white_count, empty_thr, black_thr = self._classify_warped_board(warped_board_image)
        if board is None:
            return None

        cv2.imshow("gameboard", overlay)

        if (not self._old_board_runtime_reported) or (
            (not self.is_initialized) and black_count == 12 and white_count == 12
        ):
            print("\n" + "=" * 60)
            print("BOARD STATE (OLDBOARDBETTER DEFAULT)")
            print("=" * 60)
            print(f"  Detected: Black={black_count}, White={white_count}")
            print(f"  Used thresholds: Empty<{empty_thr:.1f}<Black<{black_thr:.1f}<White")
            print(f"  Selected classifier: {self._last_selected_method}")

            if black_count == 12 and white_count == 12:
                print("  ? Perfect! Game ready")
                print("  -> Press 'S' in any OpenCV window to start the game")
                self.is_initialized = True
            else:
                print("  ? Piece count mismatch - fallback remains available if needed")

            print("=" * 60 + "\n")
            self._old_board_runtime_reported = True

        return board

    def _get_board_from_image_fallback(self, cameraImage):
        board = np.empty((8, 8), dtype=object)
        board.fill(0)

        new_image = cameraImage.copy()
        gray = cv2.cvtColor(cameraImage, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        black_count = 0
        white_count = 0
        position = 0

        for row in range(len(self.gameBoardFieldsContours)):
            for col in range(len(self.gameBoardFieldsContours[row])):
                x, y, w_rect, h_rect = self.gameBoardFieldsContours[row][col]

                padding = 10
                x_pad = x + padding
                y_pad = y + padding
                w_pad = w_rect - 2 * padding
                h_pad = h_rect - 2 * padding

                if w_pad <= 0 or h_pad <= 0:
                    board[row][col] = 0
                    position += 1
                    continue

                square = blur[y_pad:y_pad + h_pad, x_pad:x_pad + w_pad]
                if square.size == 0:
                    board[row][col] = 0
                    position += 1
                    continue

                variance = float(np.var(square))

                if variance < self.empty_variance_threshold:
                    board[row][col] = 0
                    color = (0, 255, 255)
                elif variance < self.black_variance_threshold:
                    board[row][col] = 2
                    black_count += 1
                    color = (255, 0, 255)
                else:
                    board[row][col] = 1
                    white_count += 1
                    color = (0, 255, 255)

                point_x = x + w_rect // 2
                point_y = y + h_rect // 2
                label = str(position)
                cv2.putText(new_image, label, (point_x - 10, point_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                position += 1

        cv2.putText(
            new_image,
            f"Fallback Black: {black_count} | White: {white_count}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.imshow("gameboard", new_image)

        return board

    def _trim_image_perspective(self, image, corners):
        if image is None:
            return None
        if corners is None or len(corners) != 4:
            return image

        # Match board_detection_copy warp geometry exactly.
        board_size = 800
        dst_points = np.array(
            [
                [0, 0],
                [board_size, 0],
                [board_size, board_size],
                [0, board_size],
            ],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(corners, dst_points)
        warped = cv2.warpPerspective(image, matrix, (board_size, board_size))
        return warped

    def _get_trim_param_manual(self):
        if self.ximeaCamera is None:
            return None

        corners = []
        clone = None

        def click_event(event, x, y, _flags, _params):
            nonlocal corners, clone
            if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
                corners.append([x, y])
                print(f"  ? Corner {len(corners)}/4 selected: ({x}, {y})")

                cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)
                if len(corners) > 1:
                    cv2.line(clone, tuple(corners[-2]), tuple(corners[-1]), (0, 255, 0), 2)
                if len(corners) == 4:
                    cv2.line(clone, tuple(corners[-1]), tuple(corners[0]), (0, 255, 0), 2)
                    print("\n  -> All 4 corners selected!")
                    print("  -> Press SPACE in 'Select Board Corners' window to confirm")
                    print("  -> Press 'R' to reset and reselect corners\n")
                cv2.imshow("Select Board Corners", clone)

        print("\n" + "=" * 60)
        print("STEP 1: BOARD CORNER SELECTION")
        print("=" * 60)
        print("Click on the 4 corners of the board in this order:")
        print("  1. White square on black side (Top-Left)")
        print("  2. Black square with black piece (Top-Right)")
        print("  3. White square on white side (Bottom-Right)")
        print("  4. White piece on black square (Bottom-Left)")
        print("\nControls:")
        print("  SPACE - Confirm selection")
        print("  R     - Reset points")
        print("  ESC   - Exit")
        print("-" * 60 + "\n")

        while True:
            image = self.ximeaCamera.get_camera_image()
            if image is None:
                continue
            clone = image.copy()

            for i, corner in enumerate(corners):
                cv2.circle(clone, tuple(corner), 5, (0, 255, 0), -1)
                cv2.putText(clone, str(i + 1), (corner[0] + 10, corner[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if i > 0:
                    cv2.line(clone, tuple(corners[i - 1]), tuple(corners[i]), (0, 255, 0), 2)

            if len(corners) == 4:
                cv2.line(clone, tuple(corners[-1]), tuple(corners[0]), (0, 255, 0), 2)

            cv2.imshow("Select Board Corners", clone)
            cv2.setMouseCallback("Select Board Corners", click_event)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("r"), ord("R")):
                corners = []
                print("\n  ? Points reset - start selecting again\n")

            if key == 32 and len(corners) == 4:
                cv2.destroyWindow("Select Board Corners")
                print("? Board corners saved!\n")
                print("=" * 60)
                print("STEP 2: DETECTING BOARD GRID")
                print("=" * 60 + "\n")
                return np.array(corners, dtype=np.float32)

            if key == 27:
                cv2.destroyWindow("Select Board Corners")
                print("\n? Board selection cancelled\n")
                return None

    def set_number_of_empty_fields(self, game):
        self.numberOfEmptyFields = 64 - game.board.black_left - game.board.white_left


# Compatibility alias for modules expecting board_detection.BoardDetection
BoardDetection = BoardDetector


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


def _consensus_corners(corner_candidates, bin_size: float = 8.0) -> Optional[np.ndarray]:
    if not corner_candidates:
        return None

    consensus = np.zeros((4, 2), dtype=np.float32)
    for corner_idx in range(4):
        bins = []
        point_map = {}

        for corners in corner_candidates:
            point = corners[corner_idx]
            point_key = (
                int(round(float(point[0]) / bin_size)),
                int(round(float(point[1]) / bin_size)),
            )
            bins.append(point_key)
            if point_key not in point_map:
                point_map[point_key] = np.array(point, dtype=np.float32)

        winner, _votes = Counter(bins).most_common(1)[0]
        consensus[corner_idx] = point_map[winner]

    return consensus


def _render_demo_panel(frame: np.ndarray, detector: BoardDetector, corners: Optional[np.ndarray]):
    vis = frame.copy()
    board_vis = np.zeros((800, 800, 3), dtype=np.uint8)

    if corners is not None:
        pts = corners.astype(int)
        for i in range(4):
            cv2.line(vis, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 0), 2)

        warped = detector._trim_image_perspective(frame, corners)
        board, overlay, black_count, white_count, empty_thr, black_thr = detector._classify_warped_board(warped)
        if board is not None and overlay is not None:
            board_vis = cv2.resize(overlay, (800, 800), interpolation=cv2.INTER_AREA)
            cv2.putText(board_vis, f"B:{black_count} W:{white_count}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(board_vis, f"E<{empty_thr:.1f}<B<{black_thr:.1f}<W", (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    left = cv2.resize(vis, (800, 800), interpolation=cv2.INTER_AREA)
    return np.hstack([left, board_vis])


def run_on_video(
    video_idx: int,
    slowmo: int = 120,
    hold_frames: int = 3,
    lock_mode: str = "vote",
    consensus_frames: int = 20,
):
    if video_idx < 0 or video_idx >= len(list_of_videos):
        raise IndexError(f"video_idx must be in range 0..{len(list_of_videos) - 1}")

    video_path = _resolve_video_path(list_of_videos[video_idx])
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    detector = BoardDetector(ximeaCamera=None, hold_frames=hold_frames)
    stabilizer = TemporalBoardStabilizer(hold_frames=max(0, int(hold_frames)))
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
            preview_corners, used_hold = stabilizer.update(corners)
            panel = _render_demo_panel(frame, detector, preview_corners)

            status = "DETECTED" if corners is not None else ("HOLD" if used_hold else "SEARCHING")
            color = (0, 255, 0) if status == "DETECTED" else ((0, 255, 255) if status == "HOLD" else (0, 165, 255))
            cv2.putText(panel, f"status={status} (raw)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
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

            panel = _render_demo_panel(frame, detector, preview_corners)
            if fixed_corners is not None:
                status = "LOCKED"
                color = (0, 255, 0)
            elif frame_count_for_consensus < consensus_frames:
                status = f"VOTING {frame_count_for_consensus}/{consensus_frames}"
                color = (0, 165, 255)
            else:
                status = "SEARCHING"
                color = (0, 165, 255)
            cv2.putText(panel, f"status={status} (vote)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.putText(panel, "Left: source+corners | Right: warped board recognition", (20, 770), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(panel, "Keys: q/ESC=quit, r=reset lock", (20, 795), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imshow("board_detection_1", panel)

        key = cv2.waitKey(max(1, int(slowmo))) & 0xFF
        if key in (ord("r"), ord("R")):
            fixed_corners = None
            frame_count_for_consensus = 0
            corner_candidates.clear()
            stabilizer = TemporalBoardStabilizer(hold_frames=max(0, int(hold_frames)))
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_on_ximea(
    slowmo: int = 1,
    hold_frames: int = 3,
    lock_mode: str = "vote",
    consensus_frames: int = 20,
):
    if XimeaCamera is None:
        raise RuntimeError("XimeaCamera is not available. Ensure ximea SDK/python package is installed.")

    camera = XimeaCamera()
    detector = BoardDetector(ximeaCamera=None, hold_frames=hold_frames)
    stabilizer = TemporalBoardStabilizer(hold_frames=max(0, int(hold_frames)))
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
            preview_corners, used_hold = stabilizer.update(corners)
            panel = _render_demo_panel(frame, detector, preview_corners)

            status = "DETECTED" if corners is not None else ("HOLD" if used_hold else "SEARCHING")
            color = (0, 255, 0) if status == "DETECTED" else ((0, 255, 255) if status == "HOLD" else (0, 165, 255))
            cv2.putText(panel, f"status={status} (raw)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
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

            panel = _render_demo_panel(frame, detector, preview_corners)
            if fixed_corners is not None:
                status = "LOCKED"
                color = (0, 255, 0)
            elif frame_count_for_consensus < consensus_frames:
                status = f"VOTING {frame_count_for_consensus}/{consensus_frames}"
                color = (0, 165, 255)
            else:
                status = "SEARCHING"
                color = (0, 165, 255)
            cv2.putText(panel, f"status={status} (vote)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.putText(panel, "Left: source+corners | Right: warped board recognition", (20, 770), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(panel, "Keys: q/ESC=quit, r=reset lock", (20, 795), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imshow("board_detection_1", panel)

        key = cv2.waitKey(max(1, int(slowmo))) & 0xFF
        if key in (ord("r"), ord("R")):
            fixed_corners = None
            frame_count_for_consensus = 0
            corner_candidates.clear()
            stabilizer = TemporalBoardStabilizer(hold_frames=max(0, int(hold_frames)))
        if key in (27, ord("q")):
            break

    cv2.destroyAllWindows()


def _parse_args():
    parser = argparse.ArgumentParser(description="board_detection_1: board_detection-compatible flow using old_board_better recognition")
    parser.add_argument("--mode", type=str, default="video", choices=["video", "ximea"], help="Input source")
    parser.add_argument("--video-idx", type=int, default=0, help="Index in list_of_videos for --mode video")
    parser.add_argument("--slowmo", type=int, default=120, help="Frame delay in ms")
    parser.add_argument("--lock-mode", type=str, default="vote", choices=["raw", "vote"], help="raw tracks continuously; vote locks corners after consensus")
    parser.add_argument("--hold-frames", type=int, default=3, help="How many missed frames to keep last valid board")
    parser.add_argument("--consensus-frames", type=int, default=20, help="Frames used to vote and lock corners in vote mode")
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
            hold_frames=max(0, int(args.hold_frames)),
            lock_mode=args.lock_mode,
            consensus_frames=max(1, int(args.consensus_frames)),
        )
        return

    run_on_video(
        video_idx=args.video_idx,
        slowmo=args.slowmo,
        hold_frames=max(0, int(args.hold_frames)),
        lock_mode=args.lock_mode,
        consensus_frames=max(1, int(args.consensus_frames)),
    )


if __name__ == "__main__":
    main()
