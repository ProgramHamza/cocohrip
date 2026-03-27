"""Interactive ground-truth corner labeling for videos_training clips.

Workflow:
1) Opens each video (or labeled subset) on a selected frame.
2) You click 4 corners in order: TL, TR, BR, BL.
3) Press S to save and continue.

Output JSON example:
{
  "2026-02-25_17-27-27.mp4": {
    "frame_index": 0,
    "corners": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
  }
}
"""

import argparse
import json
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Mark ground-truth board corners on videos")
    parser.add_argument("--output", type=str, default=None, help="Path to output JSON labels")
    parser.add_argument("--frame-index", type=int, default=0, help="Frame index to label in each video")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing labels for already-labeled videos")
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
    return sorted(set(videos)), repo_root


def load_labels(path: Path):
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_labels(path: Path, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)


def draw_overlay(frame, points, video_name, idx, total):
    vis = frame.copy()
    for i, (x, y) in enumerate(points):
        cv2.circle(vis, (int(x), int(y)), 6, (0, 255, 0), -1)
        cv2.putText(vis, str(i), (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if len(points) >= 2:
        for i in range(len(points) - 1):
            cv2.line(vis, tuple(map(int, points[i])), tuple(map(int, points[i + 1])), (0, 255, 0), 2)
    if len(points) == 4:
        cv2.line(vis, tuple(map(int, points[3])), tuple(map(int, points[0])), (0, 255, 0), 2)

    cv2.putText(vis, f"Video {idx}/{total}: {video_name}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(vis, "Click corners in order: TL, TR, BR, BL", (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    cv2.putText(vis, "S=save next, R=reset, N=skip, ESC=quit", (20, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    return vis


def main():
    args = parse_args()
    videos, repo_root = find_videos()
    if not videos:
        raise SystemExit("No videos found in videos_training/")

    default_output = repo_root / "src" / "checkers_game" / "checkers_game" / "camera" / "ground_truth_corners.json"
    output_path = Path(args.output) if args.output else default_output

    labels = load_labels(output_path)

    cv2.namedWindow("mark ground truth", cv2.WINDOW_NORMAL)

    for i, video_path in enumerate(videos, start=1):
        video_name = video_path.name
        if (video_name in labels) and (not args.overwrite):
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Skip (open failed): {video_name}")
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, args.frame_index))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print(f"Skip (read failed): {video_name}")
            continue

        points = []

        def on_click(event, x, y, _flags, _param):
            nonlocal points
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
                points.append((int(x), int(y)))

        cv2.setMouseCallback("mark ground truth", on_click)

        while True:
            vis = draw_overlay(frame, points, video_name, i, len(videos))
            cv2.imshow("mark ground truth", vis)
            key = cv2.waitKey(25) & 0xFF

            if key == 27:  # ESC
                save_labels(output_path, labels)
                cv2.destroyAllWindows()
                print(f"Saved partial labels to: {output_path}")
                return
            if key in (ord("r"), ord("R")):
                points = []
            if key in (ord("n"), ord("N")):
                print(f"Skipped: {video_name}")
                break
            if key in (ord("s"), ord("S")):
                if len(points) == 4:
                    labels[video_name] = {
                        "frame_index": int(args.frame_index),
                        "corners": [[int(x), int(y)] for x, y in points],
                    }
                    save_labels(output_path, labels)
                    print(f"Saved: {video_name}")
                    break
                else:
                    print("Need exactly 4 points before save.")

    cv2.destroyAllWindows()
    save_labels(output_path, labels)
    print(f"Done. Labels saved to: {output_path}")


if __name__ == "__main__":
    main()
