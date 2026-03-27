"""Combo board detector: merges old largest-quad idea with stronger line extraction.

Pipeline:
1) Grayscale + strong contrast normalization (lighting robustness)
2) Contour-focused edge map
3) Hough line extraction
4) Build line mask
5) Find largest quadrilateral whose opposite edges are parallel-ish

Run:
  PYTHONPATH=/home/marek/amavet_ws/src/checkers_game \
  python3 -m checkers_game.camera.combo --video /home/marek/amavet_ws/videos_training/2026-02-25_17-27-27.mp4 --display
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .grid_corner_detector import GridCornerDetector


class ComboBoardDetector:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        self.min_area_ratio = 0.07
        self.max_parallel_diff_deg = 18.0
        self.fallback_detector = GridCornerDetector()
        self.line_family_tolerance_deg = 12.0
        self.min_checker_score = 2.0
        self.max_board_area_ratio = 0.90
        self.min_board_area_ratio = 0.10
        self.min_border_margin_ratio = 0.02

    def detect_corners_debug(self, image: np.ndarray):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = self.clahe.apply(gray)

        # "Max" contrast normalization for robustness in darker videos.
        contrast = cv2.normalize(clahe, None, 0, 255, cv2.NORM_MINMAX)
        blur = cv2.GaussianBlur(contrast, (5, 5), 0)

        edges = cv2.Canny(blur, 45, 140)
        closed = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=2,
        )

        lines, line_stats = self._detect_lines(closed)
        line_mask = np.zeros_like(closed)
        for x1, y1, x2, y2 in lines:
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)

        # Connect segments into stronger contour candidates.
        line_mask = cv2.dilate(line_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
        line_mask = cv2.morphologyEx(
            line_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
            iterations=1,
        )

        combo_corners, quad_score = self._largest_parallel_quadrilateral(line_mask, image.shape[:2])

        # Run previous/current pipeline in parallel and choose better corners by checkerboard quality.
        fallback_result, _fallback_debug = self.fallback_detector.detect_corners_debug(image)
        fallback_corners = fallback_result.corners if fallback_result is not None else None

        combo_score = self._checkerboard_region_score(gray, combo_corners) if combo_corners is not None else -1e9
        fallback_score = self._checkerboard_region_score(gray, fallback_corners) if fallback_corners is not None else -1e9

        # Prefer fallback when it's close or better (prevents combo regressions like selecting table border).
        if fallback_corners is not None and fallback_score >= 0.90 * combo_score:
            corners = fallback_corners
            selected_source = "fallback_current"
            selected_score = fallback_score
        else:
            corners = combo_corners
            selected_source = "combo"
            selected_score = combo_score

        # Safety gate: reject implausible boards (table-border style false positives).
        is_valid = self._is_valid_board(corners, image.shape[:2], selected_score)
        if not is_valid:
            corners = None
            selected_source = "rejected"

        quad_vis = cv2.cvtColor(line_mask, cv2.COLOR_GRAY2BGR)
        if combo_corners is not None:
            pts_combo = combo_corners.astype(int)
            for i in range(4):
                cv2.line(quad_vis, tuple(pts_combo[i]), tuple(pts_combo[(i + 1) % 4]), (0, 165, 255), 1)

        if corners is not None:
            pts = corners.astype(int)
            for i in range(4):
                cv2.line(quad_vis, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 0), 2)
                cv2.putText(quad_vis, str(i), tuple(pts[i]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        debug = {
            "gray": gray,
            "contrast": contrast,
            "edges": edges,
            "closed": closed,
            "lines": lines,
            "line_mask": line_mask,
            "quad_vis": quad_vis,
            "corners": corners,
            "quad_score": quad_score,
            "line_stats": line_stats,
            "combo_score": combo_score,
            "fallback_score": fallback_score,
            "selected_source": selected_source,
            "selected_score": selected_score,
            "is_valid": bool(is_valid),
            "combo_corners": combo_corners,
            "fallback_corners": fallback_corners,
        }
        return corners, debug

    def _detect_lines(self, binary: np.ndarray):
        h, w = binary.shape[:2]
        min_len = int(0.15 * max(h, w))
        lines = cv2.HoughLinesP(
            binary,
            rho=1,
            theta=np.pi / 180,
            threshold=70,
            minLineLength=min_len,
            maxLineGap=26,
        )
        if lines is None:
            return [], {"raw": 0, "dedup": 0, "family_filtered": 0}

        raw = [tuple(map(int, l[0])) for l in lines]
        dedup = self._dedupe_lines(raw)
        filtered = self._keep_two_dominant_families(dedup)
        return filtered, {"raw": len(raw), "dedup": len(dedup), "family_filtered": len(filtered)}

    def _keep_two_dominant_families(self, lines: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Remove diagonal/noise lines by keeping only two dominant orientation families."""
        if len(lines) < 4:
            return lines

        angles = np.array([self._line_angle_deg(line) for line in lines], dtype=np.float32)

        # Orientation feature: theta and theta+180 are equivalent.
        doubled = np.deg2rad(2.0 * angles)
        features = np.column_stack((np.cos(doubled), np.sin(doubled))).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)
        _compactness, labels, _centers = cv2.kmeans(features, 2, None, criteria, 8, cv2.KMEANS_PP_CENTERS)

        fam0 = [lines[i] for i, lbl in enumerate(labels.flatten().tolist()) if int(lbl) == 0]
        fam1 = [lines[i] for i, lbl in enumerate(labels.flatten().tolist()) if int(lbl) == 1]

        def family_prune(fam):
            if not fam:
                return []
            fam_angles = np.array([self._line_angle_deg(l) for l in fam], dtype=np.float32)
            median = float(np.median(fam_angles))
            kept = [l for l, a in zip(fam, fam_angles) if self._angle_distance(float(a), median) <= self.line_family_tolerance_deg]
            return kept

        fam0 = family_prune(fam0)
        fam1 = family_prune(fam1)

        # If one family is empty after prune, fall back to original lines.
        if len(fam0) < 2 or len(fam1) < 2:
            return lines

        # Keep longer lines first inside each family.
        fam0 = sorted(fam0, key=self._line_length, reverse=True)
        fam1 = sorted(fam1, key=self._line_length, reverse=True)
        return fam0 + fam1

    def _line_angle_deg(self, line: Tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = line
        ang = math.degrees(math.atan2(float(y2 - y1), float(x2 - x1)))
        return ang + 180.0 if ang < 0 else ang

    def _line_length(self, line: Tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = line
        return float(math.hypot(x2 - x1, y2 - y1))

    def _dedupe_lines(self, lines: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        unique: List[Tuple[int, int, int, int]] = []
        for x1, y1, x2, y2 in lines:
            is_dup = False
            for ux1, uy1, ux2, uy2 in unique:
                same = abs(x1 - ux1) <= 4 and abs(y1 - uy1) <= 4 and abs(x2 - ux2) <= 4 and abs(y2 - uy2) <= 4
                rev = abs(x1 - ux2) <= 4 and abs(y1 - uy2) <= 4 and abs(x2 - ux1) <= 4 and abs(y2 - uy1) <= 4
                if same or rev:
                    is_dup = True
                    break
            if not is_dup:
                unique.append((x1, y1, x2, y2))
        return unique

    def _largest_parallel_quadrilateral(self, line_mask: np.ndarray, shape: Tuple[int, int]):
        h, w = shape
        min_area = self.min_area_ratio * h * w

        contours, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_quad = None
        best_score = -1e9

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            hull = cv2.convexHull(cnt)
            peri = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, 0.02 * peri, True)

            quad = None
            if len(approx) == 4:
                quad = approx.reshape(4, 2).astype(np.float32)
            else:
                rect = cv2.minAreaRect(hull)
                quad = cv2.boxPoints(rect).astype(np.float32)

            quad = self._order_points(quad)
            parallel_ok, parallel_score = self._parallel_constraint(quad)
            if not parallel_ok:
                continue

            # Favor large area and better parallelism.
            score = area + 15000.0 * parallel_score
            if score > best_score:
                best_score = score
                best_quad = quad

        return best_quad, best_score

    def _parallel_constraint(self, quad: np.ndarray):
        """Opposite edges should be parallel-ish under perspective (small angle difference)."""
        p = quad.reshape(4, 2)

        def edge_angle(a, b):
            v = b - a
            ang = math.degrees(math.atan2(float(v[1]), float(v[0])))
            return ang + 180.0 if ang < 0 else ang

        a0 = edge_angle(p[0], p[1])
        a1 = edge_angle(p[1], p[2])
        a2 = edge_angle(p[2], p[3])
        a3 = edge_angle(p[3], p[0])

        d02 = self._angle_distance(a0, a2)
        d13 = self._angle_distance(a1, a3)

        ok = d02 <= self.max_parallel_diff_deg and d13 <= self.max_parallel_diff_deg
        # Score in [0,1], 1 is best parallel match.
        score = 1.0 - min(1.0, (d02 + d13) / (2.0 * self.max_parallel_diff_deg + 1e-6))
        return ok, score

    def _angle_distance(self, a, b):
        d = abs(a - b) % 180.0
        return min(d, 180.0 - d)

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _checkerboard_region_score(self, gray: np.ndarray, corners: Optional[np.ndarray]) -> float:
        if corners is None:
            return -1e9
        try:
            dst = np.array([[0, 0], [320, 0], [320, 320], [0, 320]], dtype=np.float32)
            matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
            warped = cv2.warpPerspective(gray, matrix, (320, 320))
        except Exception:
            return -1e9

        cell = 40
        means = np.zeros((8, 8), dtype=np.float32)
        for r in range(8):
            for c in range(8):
                y1 = r * cell + 5
                y2 = (r + 1) * cell - 5
                x1 = c * cell + 5
                x2 = (c + 1) * cell - 5
                roi = warped[y1:y2, x1:x2]
                if roi.size == 0:
                    return -1e9
                means[r, c] = float(np.mean(roi))

        parity = (np.indices((8, 8)).sum(axis=0) % 2)
        even_vals = means[parity == 0]
        odd_vals = means[parity == 1]
        sep = abs(float(np.median(even_vals)) - float(np.median(odd_vals)))
        var = float(np.std(even_vals) + np.std(odd_vals)) + 1.0
        return sep / var

    def _is_valid_board(self, corners: Optional[np.ndarray], shape: Tuple[int, int], score: float) -> bool:
        if corners is None:
            return False
        h, w = shape
        area = self._polygon_area(corners)
        area_ratio = area / float(h * w)
        if area_ratio < self.min_board_area_ratio or area_ratio > self.max_board_area_ratio:
            return False

        pts = np.array(corners, dtype=np.float32).reshape(4, 2)
        margin = self.min_border_margin_ratio * min(h, w)
        for x, y in pts:
            if x < margin or x > (w - 1 - margin) or y < margin or y > (h - 1 - margin):
                return False

        if score < self.min_checker_score:
            return False

        return True

    def _polygon_area(self, corners: np.ndarray) -> float:
        pts = np.array(corners, dtype=np.float32).reshape(4, 2)
        x = pts[:, 0]
        y = pts[:, 1]
        return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def to_bgr(img: np.ndarray) -> np.ndarray:
    if img is None:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def make_panel(frame: np.ndarray, debug: Dict, detected: bool):
    base = frame.copy()
    gray = to_bgr(debug.get("gray"))
    contrast = to_bgr(debug.get("contrast"))
    edges = to_bgr(debug.get("edges"))
    closed = to_bgr(debug.get("closed"))
    line_mask = to_bgr(debug.get("line_mask"))
    quad_vis = to_bgr(debug.get("quad_vis"))

    line_overlay = base.copy()
    for x1, y1, x2, y2 in debug.get("lines", []):
        cv2.line(line_overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)

    result_overlay = base.copy()
    if debug.get("corners") is not None:
        pts = debug["corners"].astype(int)
        for i in range(4):
            cv2.line(result_overlay, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 0), 2)
            cv2.putText(result_overlay, str(i), tuple(pts[i]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    top = np.hstack([base, gray, contrast])
    mid = np.hstack([edges, closed, line_overlay])
    bot = np.hstack([line_mask, quad_vis, result_overlay])
    panel = np.vstack([top, mid, bot])

    status = "DETECTED" if detected else "SEARCHING"
    color = (0, 255, 0) if detected else (0, 165, 255)
    cv2.putText(panel, f"COMBO status: {status}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(panel, "Top: original | gray | max-contrast", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(panel, "Mid: edges | contour-close | Hough lines", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(panel, "Bot: line mask | biggest parallel quad | final corners", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    st = debug.get("line_stats", {})
    cv2.putText(panel, f"lines raw/dedup/family={st.get('raw',0)}/{st.get('dedup',0)}/{st.get('family_filtered',len(debug.get('lines',[])))}  quad_score={debug.get('quad_score', -1):.2f}",
                (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    cv2.putText(panel, f"source={debug.get('selected_source', 'combo')}  combo_score={debug.get('combo_score', -1):.2f}  fallback_score={debug.get('fallback_score', -1):.2f}",
                (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    cv2.putText(panel, f"selected_score={debug.get('selected_score', -1):.2f}  valid={debug.get('is_valid', False)}",
                (20, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    return panel


def fit_image_to_bounds(image: np.ndarray, max_width: int, max_height: int):
    h, w = image.shape[:2]
    if w <= max_width and h <= max_height:
        return image
    scale = min(max_width / float(w), max_height / float(h))
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)


def warp_board(image: np.ndarray, corners: np.ndarray, size: int = 800):
    dst = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(image, matrix, (size, size))


def parse_args():
    parser = argparse.ArgumentParser(description="Run combo board detector on a video")
    parser.add_argument("--video", type=str, default="/home/marek/amavet_ws/videos_training/2026-02-25_17-27-27.mp4")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--slowmo-ms", type=int, default=180)
    parser.add_argument("--display", action="store_true", default=True)
    parser.add_argument("--max-panel-width", type=int, default=1400)
    parser.add_argument("--max-panel-height", type=int, default=900)
    return parser.parse_args()


def resolve_video(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    if p.exists():
        return p
    current = Path(__file__).resolve()
    repo_root = current.parents[4]
    candidates = [repo_root / p, repo_root / "videos_training" / p.name, repo_root / "videos_training" / p.name.replace("_", "-")]
    for c in candidates:
        if c.exists():
            return c
    return p


def main():
    args = parse_args()
    video_path = resolve_video(args.video)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frames = int(max(1.0, args.seconds) * fps)

    detector = ComboBoardDetector()
    detected = None
    best_score = -1e9
    frame_idx = 0

    if args.display:
        cv2.namedWindow("combo pipeline", cv2.WINDOW_NORMAL)

    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        corners, debug = detector.detect_corners_debug(frame)

        if args.display:
            panel = make_panel(frame, debug, corners is not None)
            panel = fit_image_to_bounds(panel, args.max_panel_width, args.max_panel_height)
            cv2.imshow("combo pipeline", panel)
            key = cv2.waitKey(max(1, args.slowmo_ms)) & 0xFF
            if key == 27:
                break

        if corners is not None:
            score = float(debug.get("selected_score", -1e9))
            if score > best_score:
                best_score = score
                detected = (frame.copy(), corners.copy())

    cap.release()

    if detected is None:
        print("No corners detected within time budget")
        if args.display:
            cv2.destroyAllWindows()
        return

    print(f"Best detection score over {max_frames} frames: {best_score:.3f}")

    frame, corners = detected
    warped = warp_board(frame, corners)
    print("Corners (TL, TR, BR, BL):")
    for i, (x, y) in enumerate(corners):
        print(f"  {i}: ({x:.1f}, {y:.1f})")

    if args.display:
        result = frame.copy()
        pts = corners.astype(int)
        for i in range(4):
            cv2.line(result, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 0), 2)
            cv2.putText(result, str(i), tuple(pts[i]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("combo result", result)
        cv2.imshow("combo warped", warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
