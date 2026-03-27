"""Offline end-to-end check for board_detection_copy using recorded videos.

What it validates:
1) Corner detection and corner bounds sanity
2) Perspective warp shape consistency
3) Piece detection counts from board_detection_copy
4) AI move payload structure equivalent to CheckersNode.publish_move_state()

Usage examples:
    python -m checkers_game.camera.offline_pipeline_check --video videos_training/sample.mp4
    python -m checkers_game.camera.offline_pipeline_check --videos-dir videos_training --max-frames 240
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

from ..checkers.board import Board
from ..minimax.algorithm import minimax
from ..constants import BLACK, WHITE

try:
    from checkers_msgs.msg import Move as RosMove
    from checkers_msgs.msg import Piece as RosPiece
    ROS_MSG_AVAILABLE = True
except Exception:
    ROS_MSG_AVAILABLE = False
    RosMove = None
    RosPiece = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline pipeline checker for board detectors")
    parser.add_argument(
        "--detector",
        type=str,
        default="copy",
        choices=["copy", "bd1"],
        help="Detector backend: copy=board_detection_copy, bd1=board_detection_1",
    )
    parser.add_argument("--video", type=str, default=None, help="Single video path to test")
    parser.add_argument("--videos-dir", type=str, default="videos_training", help="Directory with training videos")
    parser.add_argument("--max-frames", type=int, default=180, help="Max frames per video")
    parser.add_argument("--ai-depth", type=int, default=3, help="Minimax depth for AI move payload test")
    parser.add_argument("--display", action="store_true", help="Show OpenCV debug windows")
    parser.add_argument("--slowmo-ms", type=int, default=1, help="Delay per frame when --display is enabled")
    parser.add_argument("--output", type=str, default=None, help="Path to JSON report")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def resolve_video_path(path_text: str) -> Path:
    raw = Path(path_text).expanduser()
    if raw.exists():
        return raw

    root = repo_root()
    candidates = [
        root / raw,
        root / "videos_training" / raw.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Video not found: {path_text}")


def discover_videos(videos_dir: str) -> list[Path]:
    raw = Path(videos_dir).expanduser()
    if raw.exists():
        base = raw
    else:
        base = repo_root() / videos_dir
    if not base.exists() or not base.is_dir():
        return []
    exts = {".mp4", ".mov", ".avi", ".mkv"}
    return sorted([p for p in base.iterdir() if p.is_file() and p.suffix.lower() in exts])


def color_to_string(color: Any) -> str:
    if color == WHITE:
        return "white"
    if color == BLACK:
        return "red"
    return "unknown"


def build_move_payload(new_move, new_piece, removed) -> dict[str, Any]:
    return {
        "target_row": int(new_move[0]),
        "target_col": int(new_move[1]),
        "piece_for_moving": {
            "row": int(new_piece.row),
            "col": int(new_piece.col),
            "color": color_to_string(new_piece.color),
            "king": bool(new_piece.king),
        },
        "removed_pieces": [
            {
                "row": int(piece.row),
                "col": int(piece.col),
                "color": color_to_string(piece.color),
                "king": bool(piece.king),
            }
            for piece in removed
        ],
    }


def validate_move_payload(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []

    for key in ("target_row", "target_col"):
        val = payload.get(key)
        if not isinstance(val, int) or not (0 <= val <= 7):
            errors.append(f"{key} invalid: {val}")

    pfm = payload.get("piece_for_moving", {})
    if not isinstance(pfm, dict):
        errors.append("piece_for_moving must be dict")
    else:
        if pfm.get("color") not in {"white", "red"}:
            errors.append(f"piece_for_moving.color invalid: {pfm.get('color')}")
        for key in ("row", "col"):
            val = pfm.get(key)
            if not isinstance(val, int) or not (0 <= val <= 7):
                errors.append(f"piece_for_moving.{key} invalid: {val}")
        if not isinstance(pfm.get("king"), bool):
            errors.append("piece_for_moving.king must be bool")

    removed = payload.get("removed_pieces")
    if not isinstance(removed, list):
        errors.append("removed_pieces must be list")
    else:
        for idx, piece in enumerate(removed):
            if not isinstance(piece, dict):
                errors.append(f"removed_pieces[{idx}] must be dict")
                continue
            if piece.get("color") not in {"white", "red"}:
                errors.append(f"removed_pieces[{idx}].color invalid: {piece.get('color')}")

    return len(errors) == 0, errors


def validate_ros_message_compatibility(payload: dict[str, Any]) -> tuple[bool, str]:
    if not ROS_MSG_AVAILABLE:
        return False, "checkers_msgs not importable in this environment"

    try:
        msg = RosMove()
        msg.target_row = payload["target_row"]
        msg.target_col = payload["target_col"]

        piece_msg = RosPiece()
        piece_msg.row = payload["piece_for_moving"]["row"]
        piece_msg.col = payload["piece_for_moving"]["col"]
        piece_msg.color = payload["piece_for_moving"]["color"]
        piece_msg.king = payload["piece_for_moving"]["king"]
        msg.piece_for_moving = piece_msg

        for removed_piece in payload["removed_pieces"]:
            rm = RosPiece()
            rm.row = removed_piece["row"]
            rm.col = removed_piece["col"]
            rm.color = removed_piece["color"]
            rm.king = removed_piece["king"]
            msg.removed_pieces.append(rm)
    except Exception as exc:
        return False, str(exc)

    return True, "ok"


def _create_detector(detector_key: str, video_path: Path, display: bool):
    if detector_key == "copy":
        from .board_detection_copy import BoardDetection as CopyBoardDetection

        return CopyBoardDetection(
            ximeaCamera=None,
            video_path=str(video_path),
            interactive=False,
            show_windows=display,
            loop_video=False,
            prefer_video=True,
        )

    if detector_key == "bd1":
        from .board_detection_1 import BoardDetection as Bd1BoardDetection

        detector = Bd1BoardDetection(ximeaCamera=None)
        if not display:
            # board_detection_1 calls cv2.imshow directly; disable display for headless runs.
            detector._offline_checker_headless = True  # marker for debugging
        return detector

    raise ValueError(f"Unsupported detector key: {detector_key}")


def run_video_check(video_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    detector = _create_detector(args.detector, video_path, args.display)
    using_bd1 = args.detector == "bd1"
    original_imshow = cv2.imshow

    cap = None
    if using_bd1:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        if not args.display:
            cv2.imshow = lambda *unused_args, **unused_kwargs: None  # noqa: E731
    else:
        if detector.video_capture is not None:
            detector.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    corners = np.array(detector.bounderies, dtype=np.float32) if detector.bounderies is not None else None
    corners_detected = corners is not None and corners.shape == (4, 2)
    corners_in_bounds = False

    game_stub = SimpleNamespace(board=SimpleNamespace(black_left=12, white_left=12))
    frames_processed = 0
    board_ok_frames = 0
    warp_ok_frames = 0
    exact_12_12_frames = 0
    black_counts: list[int] = []
    white_counts: list[int] = []

    sample_move_payload = None
    move_payload_valid = False
    move_payload_errors: list[str] = []
    ros_msg_compatible = False
    ros_msg_detail = "not_run"

    try:
        while frames_processed < max(1, int(args.max_frames)):
            if using_bd1:
                ok, frame = cap.read()
                if not ok:
                    break
            else:
                frame = detector.get_camera_image()
                if frame is None:
                    break

            frames_processed += 1

            if not corners_detected and getattr(detector, "bounderies", None) is not None:
                corners = np.array(detector.bounderies, dtype=np.float32)
                corners_detected = corners.shape == (4, 2)
            if corners_detected:
                h, w = frame.shape[:2]
                x_ok = np.logical_and(corners[:, 0] >= 0, corners[:, 0] <= w)
                y_ok = np.logical_and(corners[:, 1] >= 0, corners[:, 1] <= h)
                corners_in_bounds = bool(np.all(x_ok) and np.all(y_ok))

            warped = detector._trim_image_perspective(frame, detector.bounderies)
            if warped is not None and len(warped.shape) == 3 and warped.shape[:2] == (800, 800):
                warp_ok_frames += 1

            board_detected = detector.get_board(frame, game_stub)
            if board_detected is None:
                if args.display:
                    key = cv2.waitKey(max(1, int(args.slowmo_ms))) & 0xFF
                    if key == 27:
                        break
                continue

            board_arr = np.array(board_detected, dtype=np.int32)
            if board_arr.shape != (8, 8):
                if args.display:
                    key = cv2.waitKey(max(1, int(args.slowmo_ms))) & 0xFF
                    if key == 27:
                        break
                continue

            board_ok_frames += 1
            black_count = int(np.count_nonzero(board_arr == 2))
            white_count = int(np.count_nonzero(board_arr == 1))
            black_counts.append(black_count)
            white_counts.append(white_count)
            game_stub.board.black_left = black_count
            game_stub.board.white_left = white_count

            if black_count == 12 and white_count == 12:
                exact_12_12_frames += 1
                if sample_move_payload is None:
                    board_for_ai = Board()
                    board_for_ai.create_board(board_arr)
                    if board_for_ai.isBoardCreated:
                        _value, _new_board, new_move, new_piece, removed = minimax(
                            board_for_ai, max(1, int(args.ai_depth)), BLACK, None
                        )
                        if new_move is not None and new_piece is not None:
                            sample_move_payload = build_move_payload(new_move, new_piece, removed)
                            move_payload_valid, move_payload_errors = validate_move_payload(sample_move_payload)
                            ros_msg_compatible, ros_msg_detail = validate_ros_message_compatibility(sample_move_payload)

            if args.display:
                key = cv2.waitKey(max(1, int(args.slowmo_ms))) & 0xFF
                if key == 27:
                    break
    finally:
        if cap is not None:
            cap.release()
        if hasattr(detector, "video_capture") and detector.video_capture is not None:
            detector.video_capture.release()
        cv2.imshow = original_imshow
        if args.display:
            cv2.destroyAllWindows()

    return {
        "video_path": str(video_path),
        "frames_processed": int(frames_processed),
        "corners_detected": bool(corners_detected),
        "corners_in_bounds_0_600": bool(corners_in_bounds),
        "corners_tl_tr_br_bl": corners.tolist() if corners_detected else None,
        "warp_ok_frames": int(warp_ok_frames),
        "board_ok_frames": int(board_ok_frames),
        "exact_12_12_frames": int(exact_12_12_frames),
        "avg_black_count": float(np.mean(black_counts)) if black_counts else 0.0,
        "avg_white_count": float(np.mean(white_counts)) if white_counts else 0.0,
        "sample_move_payload": sample_move_payload,
        "move_payload_valid": bool(move_payload_valid),
        "move_payload_errors": move_payload_errors,
        "ros_message_class_available": bool(ROS_MSG_AVAILABLE),
        "ros_message_compatible": bool(ros_msg_compatible),
        "ros_message_compatibility_detail": ros_msg_detail,
    }


def main():
    args = parse_args()
    if args.video:
        videos = [resolve_video_path(args.video)]
    else:
        videos = discover_videos(args.videos_dir)
        if not videos:
            raise SystemExit(f"No videos found in: {args.videos_dir}")

    results = []
    for idx, video_path in enumerate(videos, start=1):
        print(f"[{idx}/{len(videos)}] Checking: {video_path.name}")
        try:
            result = run_video_check(video_path, args)
            print(
                "  corners={corners} board_ok={board_ok}/{frames} exact12={exact} move_payload_valid={payload}".format(
                    corners=result["corners_detected"],
                    board_ok=result["board_ok_frames"],
                    frames=result["frames_processed"],
                    exact=result["exact_12_12_frames"],
                    payload=result["move_payload_valid"],
                )
            )
        except Exception as exc:
            result = {
                "video_path": str(video_path),
                "error": str(exc),
            }
            print(f"  error: {exc}")
        results.append(result)

    output_path = (
        Path(args.output).expanduser()
        if args.output
        else repo_root()
        / "src"
        / "checkers_game"
        / "checkers_game"
        / "camera"
        / "evaluation_output"
        / f"offline_pipeline_report_{args.detector}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "settings": {
            "max_frames": int(args.max_frames),
            "ai_depth": int(args.ai_depth),
            "display": bool(args.display),
            "detector": args.detector,
            "ros_message_class_available": bool(ROS_MSG_AVAILABLE),
        },
        "videos": results,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\nSaved report: {output_path}")


if __name__ == "__main__":
    main()
