import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class DetectedGrid:
    corners: np.ndarray  # shape (4, 2), float32 ordered TL, TR, BR, BL
    intersections: List[Tuple[int, int]]
    vertical_lines: List[Tuple[int, int, int, int]]
    horizontal_lines: List[Tuple[int, int, int, int]]


class GridCornerDetector:
    """Detect board corners from a single image using line clustering and intersections.

    Steps:
        1) Preprocess (grayscale, CLAHE, blur, Canny, morphology) to highlight lines.
        2) Probabilistic Hough to get line segments.
        3) Cluster line angles into two dominant families (~orthogonal).
        4) Intersect the two families to get a grid; outer hull gives board corners.
    """

    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.parallel_angle_tolerance_deg = 18.0
        self.spacing_tolerance_ratio = 0.90
        self.target_grid_lines = 9
        self.min_required_family_lines = 7
        self.merge_angle_tolerance_deg = 4.0
        self.merge_distance_tolerance_px = 6.0
        self.thickness_score_threshold = 0.36
        self.thickness_min_keep = 18
        self.outline_min_length_ratio = 0.22
        self.outline_modulation_threshold = 0.40
        self.category_thickness_tol_px = 1.8
        self.category_angle_tol_deg = 8.0
        self.category_rho_tol_px = 14.0

    def detect_corners(self, image: np.ndarray) -> Optional[DetectedGrid]:
        result, _ = self._run_pipeline(image)
        return result

    def detect_corners_debug(self, image: np.ndarray):
        """Same as detect_corners, but also returns intermediate debug artifacts."""
        return self._run_pipeline(image)

    def _run_pipeline(self, image: np.ndarray):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe_img = self.clahe.apply(gray)
        blurred = cv2.GaussianBlur(clahe_img, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        # IMPORTANT: line extraction source is ONLY the contour-highlight image (5th debug panel).
        contour_binary = closed.copy()
        contour_binary = cv2.dilate(contour_binary, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

        raw_lines = self._detect_lines(closed)
        merged_lines = self._merge_collinear_segments(raw_lines, image.shape[:2]) if raw_lines else []
        thickness_lines, thickness_scores = self._filter_lines_by_thickness(merged_lines, contour_binary)
        lines = thickness_lines

        # Parallel reference detector (legacy-like), used for cross-check only.
        legacy_corners, legacy_score = self._legacy_reference_corners(merged_lines, image)

        selected_a = []
        selected_b = []
        intersections = []
        corners = None
        current_score = None
        selected_source = "current"
        family_stats: Dict[str, Dict[str, float]] = {}
        outline_lines = []
        outline_scores = []

        # Alternative strategy: detect 4 border lines using thickness modulation on high-contrast contour map.
        outline_result = self._detect_outline_from_thickness_categories(contour_binary, lines, image.shape[:2])
        if outline_result is not None:
            family_a, family_b, outline_scores = outline_result
            if len(family_a) >= 2 and len(family_b) >= 2:
                pair_a, pair_b, best_corners, best_outline_score = self._choose_best_outline_pairs(family_a, family_b, image, contour_binary)
                if pair_a is not None and pair_b is not None and best_corners is not None:
                    selected_a = pair_a
                    selected_b = pair_b
                    outline_lines = selected_a + selected_b
                    intersections = self._intersections(selected_a, selected_b, image.shape[:2])
                    corners = best_corners
                    current_score = float(best_outline_score)
                    outline_scores.append(float(best_outline_score))

        if lines and corners is None:
            families = self._cluster_lines(lines)
            if families is not None:
                family_a, family_b = families
                selected_a, stats_a = self._select_grid_lines(family_a)
                selected_b, stats_b = self._select_grid_lines(family_b)
                family_stats["family_a"] = stats_a
                family_stats["family_b"] = stats_b

                if len(selected_a) >= self.min_required_family_lines and len(selected_b) >= self.min_required_family_lines:
                    # Prefer 9 lines, but allow fewer when pieces/lighting hide some lines.
                    selected_a = selected_a[: self.target_grid_lines]
                    selected_b = selected_b[: self.target_grid_lines]

                    pair_a, pair_b, best_corners, _ = self._choose_best_outline_pairs(selected_a, selected_b, image, contour_binary)
                    if pair_a is not None and pair_b is not None and best_corners is not None:
                        selected_a = pair_a
                        selected_b = pair_b
                        intersections = self._intersections(selected_a, selected_b, image.shape[:2])
                        corners = best_corners
                        current_score = float(self._checkerboard_region_score(gray, best_corners))
                    else:
                        intersections = self._intersections(selected_a, selected_b, image.shape[:2])
                        corners = self._corners_from_extreme_lines(selected_a, selected_b)
                        if corners is not None:
                            current_score = float(self._checkerboard_region_score(gray, corners))

        # Prefer current detector always; use legacy only as emergency fallback.
        if corners is None and legacy_corners is not None:
            corners = legacy_corners
            intersections = []
            selected_source = "legacy-fallback"

        agreement = None
        if corners is not None and legacy_corners is not None:
            agreement = float(self._corner_agreement(corners, legacy_corners))

        result = None
        if corners is not None and (len(intersections) >= 4):
            result = DetectedGrid(
                corners=corners,
                intersections=intersections,
                vertical_lines=selected_a,
                horizontal_lines=selected_b,
            )

        debug = {
            "gray": gray,
            "clahe": clahe_img,
            "blurred": blurred,
            "edges": edges,
            "closed": closed,
            "contour_binary": contour_binary,
            "all_lines": raw_lines,
            "merged_lines": merged_lines,
            "thickness_scores": thickness_scores,
            "thickness_pass_count": len(thickness_lines),
            "outline_lines": outline_lines,
            "outline_scores": outline_scores,
            "vertical_lines": selected_a,
            "horizontal_lines": selected_b,
            "intersections": intersections,
            "corners": corners,
            "selected_source": selected_source,
            "current_score": current_score,
            "legacy_corners": legacy_corners,
            "legacy_score": legacy_score,
            "corner_agreement": agreement,
            "family_stats": family_stats,
        }
        return result, debug

    # --- pipeline steps ---

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = self.clahe.apply(gray)
        blurred = cv2.GaussianBlur(clahe, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

        # Close small gaps to make lines continuous
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        return closed

    def _detect_lines(self, edge_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect lines from contour-highlight image (white contours on black background).

        This is tuned specifically for the 5th debug panel (`closed` morphology image).
        """
        h, w = edge_image.shape[:2]
        work = edge_image.copy()
        work = cv2.dilate(work, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

        # Primary pass: longer lines, moderate gap joining
        min_len_primary = int(0.17 * max(h, w))
        lines_primary = cv2.HoughLinesP(
            work,
            rho=1,
            theta=np.pi / 180,
            threshold=90,
            minLineLength=min_len_primary,
            maxLineGap=28,
        )

        # Secondary pass: recover shorter broken segments from same contour map
        min_len_secondary = int(0.10 * max(h, w))
        lines_secondary = cv2.HoughLinesP(
            work,
            rho=1,
            theta=np.pi / 180,
            threshold=60,
            minLineLength=min_len_secondary,
            maxLineGap=20,
        )

        collected: List[Tuple[int, int, int, int]] = []
        if lines_primary is not None:
            collected.extend([tuple(map(int, line[0])) for line in lines_primary])
        if lines_secondary is not None:
            collected.extend([tuple(map(int, line[0])) for line in lines_secondary])

        if not collected:
            return []

        # Deduplicate near-identical Hough outputs by endpoint proximity.
        unique: List[Tuple[int, int, int, int]] = []
        for line in collected:
            x1, y1, x2, y2 = line
            duplicate = False
            for ux1, uy1, ux2, uy2 in unique:
                same_order = abs(x1 - ux1) <= 4 and abs(y1 - uy1) <= 4 and abs(x2 - ux2) <= 4 and abs(y2 - uy2) <= 4
                rev_order = abs(x1 - ux2) <= 4 and abs(y1 - uy2) <= 4 and abs(x2 - ux1) <= 4 and abs(y2 - uy1) <= 4
                if same_order or rev_order:
                    duplicate = True
                    break
            if not duplicate:
                unique.append(line)

        return unique

    def _build_contour_binary(self, clahe_gray: np.ndarray) -> np.ndarray:
        """Create high-contrast contour map where border thickness is preserved."""
        binary = cv2.adaptiveThreshold(
            clahe_gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            4,
        )
        # Keep bright structures as white, remove speckle noise
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        return closed

    def _detect_outline_by_thickness(self, contour_binary: np.ndarray, merged_lines: List[Tuple[int, int, int, int]], shape):
        """Detect 4 board border lines by periodic thick/thin modulation on contour map."""
        if not merged_lines:
            return None

        h, w = shape
        min_len = self.outline_min_length_ratio * max(h, w)
        scored = []

        for line in merged_lines:
            if self._line_length(line) < min_len:
                continue
            score = self._line_outline_modulation_score(line, contour_binary)
            scored.append((line, score))

        if not scored:
            return None

        # Keep only lines with strong outline modulation
        candidates = [item for item in scored if item[1] >= self.outline_modulation_threshold]
        if len(candidates) < 4:
            # fallback to strongest lines
            candidates = sorted(scored, key=lambda item: item[1], reverse=True)[:20]

        candidate_lines = [c[0] for c in candidates]
        families = self._cluster_lines(candidate_lines)
        if families is None:
            return None

        fam_a, fam_b = families
        if len(fam_a) < 2 or len(fam_b) < 2:
            return None

        # Pick two extreme lines in each family -> outline borders
        fam_a_sorted = self._sort_family_by_position(fam_a)
        fam_b_sorted = self._sort_family_by_position(fam_b)
        selected_a = [fam_a_sorted[0], fam_a_sorted[-1]]
        selected_b = [fam_b_sorted[0], fam_b_sorted[-1]]
        scores_only = [float(s) for _, s in candidates]
        return selected_a, selected_b, scores_only

    def _detect_outline_from_thickness_categories(self, contour_binary: np.ndarray, lines: List[Tuple[int, int, int, int]], shape):
        """Category-based outline detection requested by user.

        Steps:
            1) Estimate thickness for every detected line.
            2) Extend lines to image bounds (infinite-line interpretation).
            3) Merge overlapping lines with similar thickness into one category.
            4) If one thickness category yields at least 4 merged lines, split into 2 families,
               take extremes (2 + 2) as board outline.
        """
        if not lines:
            return None

        h, w = shape

        # 1) Build enriched line descriptors
        descriptors = []
        for line in lines:
            angle = self._line_angle_deg(line)
            angle_rad = math.radians(angle)
            direction = np.array([math.cos(angle_rad), math.sin(angle_rad)], dtype=np.float32)
            normal = np.array([-direction[1], direction[0]], dtype=np.float32)
            x1, y1, x2, y2 = line
            midpoint = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
            rho = float(np.dot(midpoint, normal))
            thickness = self._estimate_line_thickness(line, contour_binary)
            ext = self._extend_line_to_bounds(line, w, h)
            descriptors.append(
                {
                    "line": line,
                    "extended": ext,
                    "angle": angle,
                    "rho": rho,
                    "thickness": thickness,
                }
            )

        if len(descriptors) < 4:
            return None

        # 2) Merge overlapping + same-thickness descriptors into categories (one physical line)
        categories = []
        for d in descriptors:
            placed = False
            for cat in categories:
                if self._descriptor_matches_category(d, cat):
                    cat.append(d)
                    placed = True
                    break
            if not placed:
                categories.append([d])

        merged_lines = []
        merged_thickness = []
        for cat in categories:
            rep = self._merge_descriptor_category(cat, w, h)
            if rep is not None:
                merged_lines.append(rep["line"])
                merged_thickness.append(rep["thickness"])

        if len(merged_lines) < 4:
            return None

        # 3) Thickness categories: lines with similar thickness go together
        thickness_groups: List[List[Tuple[int, int, int, int]]] = []
        thickness_values: List[float] = []
        for line, t in zip(merged_lines, merged_thickness):
            assigned = False
            for idx, ref_t in enumerate(thickness_values):
                if abs(t - ref_t) <= self.category_thickness_tol_px:
                    thickness_groups[idx].append(line)
                    # update reference thickness slowly
                    thickness_values[idx] = 0.7 * thickness_values[idx] + 0.3 * t
                    assigned = True
                    break
            if not assigned:
                thickness_values.append(t)
                thickness_groups.append([line])

        # 4) Find any category that can produce outline (>=4 lines)
        best = None
        best_count = -1
        for group in thickness_groups:
            if len(group) < 4:
                continue
            fams = self._cluster_lines(group)
            if fams is None:
                continue
            fam_a, fam_b = fams
            if len(fam_a) < 2 or len(fam_b) < 2:
                continue
            if len(group) > best_count:
                best_count = len(group)
                best = (fams, group)

        if best is None:
            return None

        (fam_a, fam_b), group = best
        # outline scores: category sizes/diagnostic
        scores = [float(len(g)) for g in thickness_groups]
        return fam_a, fam_b, scores

    def _choose_best_outline_pairs(self, family_a, family_b, image: np.ndarray, contour_binary: np.ndarray):
        """Choose border lines by scoring candidate quadrilaterals against checkerboard structure.

        This prevents selecting table edges when they are stronger than board edges.
        """
        family_a = self._sort_family_by_position(family_a)
        family_b = self._sort_family_by_position(family_b)

        if len(family_a) < 2 or len(family_b) < 2:
            return None, None, None, -1e9

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        def pair_candidates(lines):
            cands = []
            n = len(lines)
            for i in range(n - 1):
                for j in range(i + 1, n):
                    if j - i < max(1, n // 4):
                        continue
                    cands.append((lines[i], lines[j]))
            if not cands:
                cands = [(lines[0], lines[-1])]
            return cands

        pairs_a = pair_candidates(family_a)
        pairs_b = pair_candidates(family_b)

        best_score = -1e9
        best = (None, None, None)

        for pa in pairs_a:
            for pb in pairs_b:
                corners = self._corners_from_extreme_lines(pa, pb)
                if corners is None:
                    continue

                area = self._polygon_area(corners)
                if area < 0.06 * gray.shape[0] * gray.shape[1]:
                    continue

                score = self._checkerboard_region_score(gray, corners)
                border_support = self._border_support_score(contour_binary, corners)
                margin_penalty = self._image_margin_penalty(corners, gray.shape[1], gray.shape[0])

                # prefer larger, board-like quadrilateral
                score += 0.00002 * area
                score += 1.35 * border_support
                score -= 0.85 * margin_penalty

                if score > best_score:
                    best_score = score
                    best = (list(pa), list(pb), corners)

        return best[0], best[1], best[2], best_score

    def _checkerboard_region_score(self, gray: np.ndarray, corners: np.ndarray) -> float:
        try:
            dst = np.array([[0, 0], [320, 0], [320, 320], [0, 320]], dtype=np.float32)
            matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
            warped = cv2.warpPerspective(gray, matrix, (320, 320))
        except Exception:
            return -1e6

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
                    return -1e6
                means[r, c] = float(np.mean(roi))

        parity = (np.indices((8, 8)).sum(axis=0) % 2)
        even_vals = means[parity == 0]
        odd_vals = means[parity == 1]

        med_even = float(np.median(even_vals))
        med_odd = float(np.median(odd_vals))
        sep = abs(med_even - med_odd)

        var_even = float(np.std(even_vals))
        var_odd = float(np.std(odd_vals))
        contrast_score = sep / (var_even + var_odd + 1.0)

        # Explicit checker fit quality: how well 8x8 cells match alternating two-level model.
        expected = np.where(parity == 0, med_even, med_odd)
        fit_error = float(np.mean(np.abs(means - expected)))
        fit_score = sep / (fit_error + 1.0)

        # Boundary periodicity (grid should induce repeated transitions along x and y).
        proj_x = np.mean(warped, axis=0).astype(np.float32)
        proj_y = np.mean(warped, axis=1).astype(np.float32)
        per_x = self._periodic_projection_score(proj_x)
        per_y = self._periodic_projection_score(proj_y)

        return 0.45 * contrast_score + 0.35 * fit_score + 0.20 * (per_x + per_y)

    def _periodic_projection_score(self, signal: np.ndarray) -> float:
        signal = signal - float(np.mean(signal))
        std = float(np.std(signal))
        if std < 1e-6:
            return 0.0
        signal = signal / std

        n = len(signal)
        # 8 cells in 320px -> expected periodicity ~40px; allow broad neighborhood.
        min_lag = max(8, n // 12)
        max_lag = min(n // 4, n - 2)
        if max_lag <= min_lag:
            return 0.0

        best = 0.0
        for lag in range(min_lag, max_lag + 1):
            a = signal[:-lag]
            b = signal[lag:]
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            if denom <= 1e-6:
                continue
            corr = float(np.dot(a, b) / denom)
            if corr > best:
                best = corr
        return max(0.0, best)

    def _border_support_score(self, contour_binary: np.ndarray, corners: np.ndarray) -> float:
        """How strongly warped border aligns with bright contour pixels (board outline evidence)."""
        try:
            dst = np.array([[0, 0], [320, 0], [320, 320], [0, 320]], dtype=np.float32)
            matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
            warped = cv2.warpPerspective(contour_binary, matrix, (320, 320))
        except Exception:
            return 0.0

        warped = (warped > 0).astype(np.float32)
        b = 10
        top = float(np.mean(warped[0:b, :]))
        bottom = float(np.mean(warped[-b:, :]))
        left = float(np.mean(warped[:, 0:b]))
        right = float(np.mean(warped[:, -b:]))
        border = (top + bottom + left + right) / 4.0

        inner = float(np.mean(warped[b:-b, b:-b])) if warped.shape[0] > 2 * b and warped.shape[1] > 2 * b else 0.0
        # Want strong border and not uniformly dense everywhere.
        return max(0.0, border - 0.35 * inner)

    def _image_margin_penalty(self, corners: np.ndarray, width: int, height: int) -> float:
        """Penalty when candidate corners hug the frame border (common for table edge)."""
        pts = corners.reshape(-1, 2)
        margin = max(8.0, 0.02 * min(width, height))
        penalties = []
        for x, y in pts:
            dx = min(float(x), float(width - 1 - x))
            dy = min(float(y), float(height - 1 - y))
            d = min(dx, dy)
            if d >= margin:
                penalties.append(0.0)
            else:
                penalties.append((margin - d) / margin)
        return float(np.mean(penalties))

    def _polygon_area(self, corners: np.ndarray) -> float:
        pts = corners.reshape(-1, 2)
        x = pts[:, 0]
        y = pts[:, 1]
        return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    def _legacy_reference_corners(self, merged_lines: List[Tuple[int, int, int, int]], image: np.ndarray):
        """Legacy-style reference path for comparison (not primary decision)."""
        if len(merged_lines) < 4:
            return None, None

        families = self._cluster_lines(merged_lines)
        if families is None:
            return None, None

        fam_a, fam_b = families
        if len(fam_a) < 2 or len(fam_b) < 2:
            return None, None

        fam_a = self._sort_family_by_position(fam_a)
        fam_b = self._sort_family_by_position(fam_b)
        corners = self._corners_from_extreme_lines([fam_a[0], fam_a[-1]], [fam_b[0], fam_b[-1]])
        if corners is None:
            return None, None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        score = float(self._checkerboard_region_score(gray, corners))
        return corners, score

    def _corner_agreement(self, c1: np.ndarray, c2: np.ndarray) -> float:
        """Agreement in [0,1], higher means corner sets are closer."""
        a = np.array(c1, dtype=np.float32).reshape(4, 2)
        b = np.array(c2, dtype=np.float32).reshape(4, 2)
        dist = np.linalg.norm(a - b, axis=1)
        mean_dist = float(np.mean(dist))
        diag = float(math.hypot(float(np.max(a[:, 0]) - np.min(a[:, 0])), float(np.max(a[:, 1]) - np.min(a[:, 1])))) + 1e-6
        normalized = mean_dist / diag
        return max(0.0, 1.0 - normalized)

    def _descriptor_matches_category(self, descriptor: Dict, category: List[Dict]) -> bool:
        angles = [c["angle"] for c in category]
        rhos = [c["rho"] for c in category]
        thicks = [c["thickness"] for c in category]

        ref_angle = float(np.median(angles))
        ref_rho = float(np.median(rhos))
        ref_thickness = float(np.median(thicks))

        return (
            self._angle_distance(descriptor["angle"], ref_angle) <= self.category_angle_tol_deg
            and abs(descriptor["rho"] - ref_rho) <= self.category_rho_tol_px
            and abs(descriptor["thickness"] - ref_thickness) <= self.category_thickness_tol_px
        )

    def _merge_descriptor_category(self, category: List[Dict], width: int, height: int):
        if not category:
            return None
        if len(category) == 1:
            return {"line": category[0]["extended"], "thickness": category[0]["thickness"]}

        angle = float(np.median([c["angle"] for c in category]))
        rho = float(np.median([c["rho"] for c in category]))
        thickness = float(np.median([c["thickness"] for c in category]))

        angle_rad = math.radians(angle)
        direction = np.array([math.cos(angle_rad), math.sin(angle_rad)], dtype=np.float32)
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)
        p0 = normal * rho
        diag = float(math.hypot(width, height))
        p1 = p0 - direction * diag
        p2 = p0 + direction * diag
        pt1 = (int(round(float(p1[0]))), int(round(float(p1[1]))))
        pt2 = (int(round(float(p2[0]))), int(round(float(p2[1]))))
        ok, c1, c2 = cv2.clipLine((0, 0, width, height), pt1, pt2)
        if not ok:
            return None
        return {"line": (c1[0], c1[1], c2[0], c2[1]), "thickness": thickness}

    def _extend_line_to_bounds(self, line: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = line
        direction = np.array([x2 - x1, y2 - y1], dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return line
        direction /= norm
        p0 = np.array([x1, y1], dtype=np.float32)
        diag = float(math.hypot(width, height))
        p1 = p0 - direction * diag
        p2 = p0 + direction * diag
        pt1 = (int(round(float(p1[0]))), int(round(float(p1[1]))))
        pt2 = (int(round(float(p2[0]))), int(round(float(p2[1]))))
        ok, c1, c2 = cv2.clipLine((0, 0, width, height), pt1, pt2)
        if not ok:
            return line
        return (c1[0], c1[1], c2[0], c2[1])

    def _estimate_line_thickness(self, line: Tuple[int, int, int, int], contour_binary: np.ndarray) -> float:
        x1, y1, x2, y2 = line
        length = self._line_length(line)
        if length < 25:
            return 0.0

        direction = np.array([x2 - x1, y2 - y1], dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return 0.0
        direction /= norm
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)

        h, w = contour_binary.shape[:2]
        samples = min(120, max(30, int(length // 3)))
        half_scan = 10
        runs = []

        for t in np.linspace(0.10, 0.90, samples):
            px = x1 + t * (x2 - x1)
            py = y1 + t * (y2 - y1)
            run = 0
            for d in range(-half_scan, half_scan + 1):
                sx = int(round(px + d * normal[0]))
                sy = int(round(py + d * normal[1]))
                if 0 <= sx < w and 0 <= sy < h and contour_binary[sy, sx] > 0:
                    run += 1
            if run > 0:
                runs.append(float(run))

        if not runs:
            return 0.0
        return float(np.median(np.array(runs, dtype=np.float32)))

    def _line_outline_modulation_score(self, line: Tuple[int, int, int, int], contour_binary: np.ndarray) -> float:
        """Score line by repeated thick/thin contour pattern along the line."""
        x1, y1, x2, y2 = line
        length = self._line_length(line)
        if length < 40:
            return 0.0

        direction = np.array([x2 - x1, y2 - y1], dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return 0.0
        direction /= norm
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)

        h, w = contour_binary.shape[:2]
        samples = min(150, max(50, int(length // 2)))
        half_scan = 8
        values = []

        for t in np.linspace(0.08, 0.92, samples):
            px = x1 + t * (x2 - x1)
            py = y1 + t * (y2 - y1)

            run = 0
            for d in range(-half_scan, half_scan + 1):
                sx = int(round(px + d * normal[0]))
                sy = int(round(py + d * normal[1]))
                if 0 <= sx < w and 0 <= sy < h and contour_binary[sy, sx] > 0:
                    run += 1
            values.append(float(run))

        if len(values) < 10:
            return 0.0

        profile = np.array(values, dtype=np.float32)
        mean = float(np.mean(profile))
        std = float(np.std(profile))
        if mean < 0.5:
            return 0.0

        # Repetition: count local peaks in thickness profile
        peaks = 0
        for i in range(1, len(profile) - 1):
            if profile[i] > profile[i - 1] and profile[i] > profile[i + 1] and profile[i] > mean + 0.35 * std:
                peaks += 1

        cv_ratio = std / (mean + 1e-6)
        peak_factor = min(1.0, peaks / 8.0)
        return 0.55 * min(1.0, cv_ratio) + 0.45 * peak_factor

    def _cluster_lines(self, lines: List[Tuple[int, int, int, int]]):
        """Split lines into two dominant orientation families using k-means on doubled angle."""
        if len(lines) < 4:
            return None

        angles = np.array([self._line_angle_deg(line) for line in lines], dtype=np.float32)

        # Orientation-aware feature: theta and theta+180 are the same line direction.
        doubled = np.deg2rad(angles * 2.0)
        features = np.column_stack((np.cos(doubled), np.sin(doubled))).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)
        _compactness, labels, _centers = cv2.kmeans(features, 2, None, criteria, 8, cv2.KMEANS_PP_CENTERS)

        families = ([], [])
        for line, lbl in zip(lines, labels.flatten().tolist()):
            families[int(lbl)].append(line)

        # Ensure both groups present
        v_lines, h_lines = families
        if len(v_lines) < 2 or len(h_lines) < 2:
            return None

        v_lines = self._sort_family_by_position(v_lines)
        h_lines = self._sort_family_by_position(h_lines)
        return v_lines, h_lines

    def _select_grid_lines(self, lines: List[Tuple[int, int, int, int]]):
        """Keep near-parallel lines and select a robust lattice-consistent subset."""
        if len(lines) < self.min_required_family_lines:
            return [], {"count": 0.0, "input_count": float(len(lines)), "mean_gap": -1.0, "std_gap": -1.0}

        angles = np.array([self._line_angle_deg(l) for l in lines], dtype=np.float32)
        median_angle = float(np.median(angles))

        # Keep near-parallel lines with adaptive tolerance (perspective introduces angle spread).
        angle_dist = np.array([self._angle_distance(float(a), median_angle) for a in angles], dtype=np.float32)
        mad = float(np.median(np.abs(angle_dist - np.median(angle_dist))))
        adaptive_tol = max(self.parallel_angle_tolerance_deg, 3.0 * mad + 2.0)

        # Keep only near-parallel lines
        parallel = [
            line
            for line, angle in zip(lines, angles)
            if self._angle_distance(float(angle), median_angle) <= adaptive_tol
        ]

        if len(parallel) < self.min_required_family_lines:
            return [], {
                "count": 0.0,
                "input_count": float(len(lines)),
                "parallel_count": float(len(parallel)),
                "mean_gap": -1.0,
                "std_gap": -1.0,
                "median_angle": median_angle,
                "adaptive_tol": adaptive_tol,
            }

        # Build normal direction for line-position measurements
        angle_rad = math.radians(median_angle)
        normal = np.array([-math.sin(angle_rad), math.cos(angle_rad)], dtype=np.float32)

        with_pos = []
        for line in parallel:
            x1, y1, x2, y2 = line
            midpoint = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
            position = float(np.dot(midpoint, normal))
            with_pos.append((line, position))

        # Deduplicate very close lines (same board line detected multiple times)
        with_pos.sort(key=lambda item: item[1])
        deduped = []
        min_sep = 6.0
        for line, pos in with_pos:
            if not deduped or abs(pos - deduped[-1][1]) > min_sep:
                deduped.append((line, pos))
            else:
                # keep longer line
                old_line = deduped[-1][0]
                if self._line_length(line) > self._line_length(old_line):
                    deduped[-1] = (line, pos)

        if len(deduped) < self.min_required_family_lines:
            return [], {
                "count": 0.0,
                "input_count": float(len(lines)),
                "parallel_count": float(len(parallel)),
                "dedup_count": float(len(deduped)),
                "mean_gap": -1.0,
                "std_gap": -1.0,
                "median_angle": median_angle,
                "adaptive_tol": adaptive_tol,
            }

        # Lattice snapping: infer pitch from positions and keep one line per snapped index.
        dedup_positions = np.array([d[1] for d in deduped], dtype=np.float32)
        gaps = np.diff(dedup_positions)
        positive_gaps = gaps[gaps > 0]
        if len(positive_gaps) == 0:
            return [], {
                "count": 0.0,
                "input_count": float(len(lines)),
                "parallel_count": float(len(parallel)),
                "dedup_count": float(len(deduped)),
                "mean_gap": -1.0,
                "std_gap": -1.0,
                "median_angle": median_angle,
                "adaptive_tol": adaptive_tol,
            }

        base_pitch = float(np.median(positive_gaps))
        candidate_pitches = [base_pitch]
        candidate_pitches.append(max(2.0, base_pitch * 0.85))
        candidate_pitches.append(base_pitch * 1.15)

        best_lines = []
        best_mean = -1.0
        best_std = -1.0
        best_score = float("inf")

        for pitch in candidate_pitches:
            for anchor in dedup_positions:
                index_map: Dict[int, Tuple[Tuple[int, int, int, int], float, float]] = {}
                errors = []

                for line, pos in deduped:
                    idx = int(round((pos - float(anchor)) / pitch))
                    snapped = float(anchor) + idx * pitch
                    err = abs(pos - snapped)
                    length = self._line_length(line)

                    existing = index_map.get(idx)
                    if existing is None or err < existing[1] or (abs(err - existing[1]) < 1e-6 and length > existing[2]):
                        index_map[idx] = (line, err, length)

                if len(index_map) < self.min_required_family_lines:
                    continue

                sorted_items = sorted(index_map.items(), key=lambda kv: kv[0])
                items = [item[1] for item in sorted_items]
                lines_for_anchor = [it[0] for it in items]
                errs_for_anchor = np.array([it[1] for it in items], dtype=np.float32)

                # If too many lines, keep best target_grid_lines by local window on indices.
                if len(lines_for_anchor) > self.target_grid_lines:
                    best_window = None
                    best_window_score = float("inf")
                    for start in range(0, len(items) - self.target_grid_lines + 1):
                        window = items[start : start + self.target_grid_lines]
                        w_err = np.array([w[1] for w in window], dtype=np.float32)
                        w_len = float(sum(w[2] for w in window))
                        w_score = float(np.mean(w_err)) - 0.0001 * w_len
                        if w_score < best_window_score:
                            best_window_score = w_score
                            best_window = window
                    if best_window is not None:
                        lines_for_anchor = [w[0] for w in best_window]
                        errs_for_anchor = np.array([w[1] for w in best_window], dtype=np.float32)

                mean_err = float(np.mean(errs_for_anchor))
                std_err = float(np.std(errs_for_anchor))
                total_length = float(sum(self._line_length(line) for line in lines_for_anchor))
                score = mean_err + 0.5 * std_err - 0.00015 * total_length

                if score < best_score:
                    best_score = score
                    best_lines = lines_for_anchor
                    best_mean = mean_err
                    best_std = std_err

        if not best_lines:
            return [], {
                "count": 0.0,
                "input_count": float(len(lines)),
                "parallel_count": float(len(parallel)),
                "dedup_count": float(len(deduped)),
                "mean_gap": -1.0,
                "std_gap": -1.0,
                "median_angle": median_angle,
                "adaptive_tol": adaptive_tol,
            }

        best_lines = self._sort_family_by_position(best_lines)
        return best_lines, {
            "count": float(len(best_lines)),
            "input_count": float(len(lines)),
            "parallel_count": float(len(parallel)),
            "dedup_count": float(len(deduped)),
            "mean_gap": best_mean,
            "std_gap": best_std,
            "median_angle": median_angle,
            "adaptive_tol": adaptive_tol,
        }

    def _intersections(self, v_lines, h_lines, shape) -> List[Tuple[int, int]]:
        h, w = shape
        points = []
        for vx1, vy1, vx2, vy2 in v_lines:
            for hx1, hy1, hx2, hy2 in h_lines:
                pt = self._line_intersection((vx1, vy1, vx2, vy2), (hx1, hy1, hx2, hy2))
                if pt is None:
                    continue
                x, y = pt
                if 0 <= x < w and 0 <= y < h:
                    points.append((int(x), int(y)))
        return points

    def _outer_corners_from_points(self, points: List[Tuple[int, int]]) -> Optional[np.ndarray]:
        pts = np.array(points, dtype=np.float32)
        if len(pts) < 4:
            return None

        # Use minAreaRect to get a stable bounding quad
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect)  # returns 4 points
        box = np.array(box, dtype=np.float32)
        return self._order_points(box)

    def _corners_from_extreme_lines(self, family_a, family_b) -> Optional[np.ndarray]:
        if len(family_a) < 2 or len(family_b) < 2:
            return None

        family_a = self._sort_family_by_position(family_a)
        family_b = self._sort_family_by_position(family_b)

        a0, a8 = family_a[0], family_a[-1]
        b0, b8 = family_b[0], family_b[-1]

        p1 = self._line_intersection(a0, b0)
        p2 = self._line_intersection(a8, b0)
        p3 = self._line_intersection(a8, b8)
        p4 = self._line_intersection(a0, b8)

        if p1 is None or p2 is None or p3 is None or p4 is None:
            return None

        corners = np.array([p1, p2, p3, p4], dtype=np.float32)
        return self._order_points(corners)

    # --- helpers ---

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points as TL, TR, BR, BL based on sums/diffs."""
        rect = np.zeros((4, 2), dtype=np.float32)

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # TL
        rect[2] = pts[np.argmax(s)]  # BR

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # TR
        rect[3] = pts[np.argmax(diff)]  # BL
        return rect

    def _angle_distance(self, a, b):
        d = abs(a - b) % 180.0
        return min(d, 180.0 - d)

    def _line_intersection(self, line1, line2):
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            return None

        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        return px, py

    def _line_angle_deg(self, line: Tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = line
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        return angle + 180.0 if angle < 0 else angle

    def _line_length(self, line: Tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = line
        return float(math.hypot(x2 - x1, y2 - y1))

    def _sort_family_by_position(self, lines: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        if not lines:
            return lines
        angles = np.array([self._line_angle_deg(l) for l in lines], dtype=np.float32)
        median_angle = float(np.median(angles))
        angle_rad = math.radians(median_angle)
        normal = np.array([-math.sin(angle_rad), math.cos(angle_rad)], dtype=np.float32)

        def line_position(line):
            x1, y1, x2, y2 = line
            midpoint = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
            return float(np.dot(midpoint, normal))

        return sorted(lines, key=line_position)

    def _merge_collinear_segments(self, lines: List[Tuple[int, int, int, int]], shape) -> List[Tuple[int, int, int, int]]:
        """Merge short line fragments that belong to the same underlying line.

        Grouping logic:
            - similar orientation (angle tolerance)
            - close normal distance (rho in normal form)
        Then each group is converted to one long clipped segment across image bounds.
        """
        if len(lines) < 2:
            return lines

        h, w = shape
        image_diag = float(math.hypot(w, h))

        params = []
        for line in lines:
            x1, y1, x2, y2 = line
            angle = self._line_angle_deg(line)
            angle_rad = math.radians(angle)
            direction = np.array([math.cos(angle_rad), math.sin(angle_rad)], dtype=np.float32)
            normal = np.array([-direction[1], direction[0]], dtype=np.float32)
            midpoint = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
            rho = float(np.dot(midpoint, normal))
            params.append((line, angle, direction, normal, rho))

        groups: List[List[Tuple[Tuple[int, int, int, int], float, np.ndarray, np.ndarray, float]]] = []
        for item in params:
            line, angle, _, _, rho = item
            placed = False

            for group in groups:
                group_angles = [g[1] for g in group]
                group_rhos = [g[4] for g in group]
                mean_angle = float(np.median(group_angles))
                mean_rho = float(np.median(group_rhos))

                if self._angle_distance(angle, mean_angle) <= self.merge_angle_tolerance_deg and abs(rho - mean_rho) <= self.merge_distance_tolerance_px:
                    group.append(item)
                    placed = True
                    break

            if not placed:
                groups.append([item])

        merged = []
        for group in groups:
            if len(group) == 1:
                merged.append(group[0][0])
                continue

            group_angles = [g[1] for g in group]
            group_rhos = [g[4] for g in group]
            avg_angle = float(np.median(group_angles))
            avg_rho = float(np.median(group_rhos))

            angle_rad = math.radians(avg_angle)
            direction = np.array([math.cos(angle_rad), math.sin(angle_rad)], dtype=np.float32)
            normal = np.array([-direction[1], direction[0]], dtype=np.float32)

            # Point on infinite line in normal form: n.x = rho
            p0 = normal * avg_rho
            p1 = p0 - direction * image_diag
            p2 = p0 + direction * image_diag

            pt1 = (int(round(float(p1[0]))), int(round(float(p1[1]))))
            pt2 = (int(round(float(p2[0]))), int(round(float(p2[1]))))

            ok, c1, c2 = cv2.clipLine((0, 0, w, h), pt1, pt2)
            if ok:
                merged.append((c1[0], c1[1], c2[0], c2[1]))

        return merged if merged else lines

    def _filter_lines_by_thickness(self, lines: List[Tuple[int, int, int, int]], contour_binary: np.ndarray):
        """Filter merged lines by repeated thick/thin contour pattern along the line.

        Requested order:
            1) detect lines
            2) project onto high-contrast contour map
            3) keep lines with periodic thickness modulation
        """
        if not lines:
            return [], []

        scored = []
        for line in lines:
            score = self._line_outline_modulation_score(line, contour_binary)
            scored.append((line, score))

        passed = [line for line, score in scored if score >= self.thickness_score_threshold]

        scores = [score for _, score in scored]

        # Avoid dropping too many lines when contour extraction is noisy.
        if len(passed) < self.thickness_min_keep and len(scored) >= self.thickness_min_keep:
            scored_sorted = sorted(scored, key=lambda item: item[1], reverse=True)
            passed = [line for line, _ in scored_sorted[: self.thickness_min_keep]]

        return passed, scores


def warp_board(image: np.ndarray, corners: np.ndarray, size: int = 800) -> np.ndarray:
    """Warp image to a square top-down board view using detected corners."""
    dst = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(image, matrix, (size, size))
