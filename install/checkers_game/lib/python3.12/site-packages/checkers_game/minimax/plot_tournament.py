import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - needed to register 3D projection


def load_csv(path: Path) -> List[dict]:
    with path.open() as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def load_ranking(path: Path) -> List[Tuple[str, float]]:
    rows = load_csv(path)
    ranking = []
    for row in rows:
        try:
            score = float(row.get("score", 0))
        except ValueError:
            score = 0.0
        ranking.append((row.get("name", ""), score))
    return ranking


def build_heatmap(series_rows: List[dict], phases: List[str]) -> Tuple[List[str], List[List[float]]]:
    names = set()
    for row in series_rows:
        if row.get("phase") in phases:
            names.add(row.get("name_a", ""))
            names.add(row.get("name_b", ""))
    ordered = sorted([n for n in names if n])
    idx = {name: i for i, name in enumerate(ordered)}
    size = len(ordered)
    matrix = [[0.0 for _ in range(size)] for _ in range(size)]

    for row in series_rows:
        if row.get("phase") not in phases:
            continue
        a = row.get("name_a", "")
        b = row.get("name_b", "")
        if not a or not b:
            continue
        try:
            pa = float(row.get("points_a", 0))
            pb = float(row.get("points_b", 0))
        except ValueError:
            pa = pb = 0.0
        matrix[idx[a]][idx[b]] = pa
        matrix[idx[b]][idx[a]] = pb
    return ordered, matrix


def plot_heatmap(names: List[str], matrix: List[List[float]], out_path: Path, title: str) -> None:
    n = len(names)
    fig = plt.figure(figsize=(max(7, n * 0.7), max(6, n * 0.7)))
    ax = fig.add_subplot(111, projection="3d")

    xs, ys, zs, values = [], [], [], []
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            xs.append(i)
            ys.append(j)
            zs.append(0)
            values.append(val)

    dx = dy = 0.6
    ax.bar3d(xs, ys, zs, dx, dy, values, shade=True, color="#4e79a7", alpha=0.9)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_zlabel("Points")
    ax.set_title(title)

    plt.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_ranking_bar(ranking: List[Tuple[str, float]], out_path: Path) -> None:
    if not ranking:
        return
    names = [r[0] for r in ranking]
    scores = [r[1] for r in ranking]
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.5), 4))
    ax.bar(range(len(names)), scores, color="#4e79a7")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Total points")
    ax.set_title("Tournament ranking")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def compute_alternating_winrates(game_rows: List[dict], eval_games_per_round: int) -> Tuple[List[int], Dict[str, List[float]]]:
    eval_rows = [r for r in game_rows if r.get("phase") == "alternating_eval"]
    if not eval_rows:
        return [], {}
    # assume single pair; pair format "A_vs_B"
    pair = eval_rows[0].get("pair", "")
    if "_vs_" not in pair:
        return [], {}
    name_a, name_b = pair.split("_vs_", 1)
    round_count = len(eval_rows) // max(1, eval_games_per_round)
    rounds = []
    winrates: Dict[str, List[float]] = {name_a: [], name_b: []}
    for r in range(round_count):
        chunk = eval_rows[r * eval_games_per_round : (r + 1) * eval_games_per_round]
        scores = defaultdict(float)
        for g in chunk:
            winner = g.get("winner")
            black = g.get("black")
            white = g.get("white")
            if winner == "black":
                scores[black] += 1.0
            elif winner == "white":
                scores[white] += 1.0
            else:
                scores[black] += 0.5
                scores[white] += 0.5
        total = len(chunk)
        rounds.append(r + 1)
        winrates[name_a].append(scores[name_a] / total if total else 0.0)
        winrates[name_b].append(scores[name_b] / total if total else 0.0)
    return rounds, winrates


def plot_alternating_eval(rounds: List[int], winrates: Dict[str, List[float]], out_path: Path) -> None:
    if not rounds or not winrates:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, wr in winrates.items():
        ax.plot(rounds, wr, marker="o", label=name)
    ax.set_xlabel("Round")
    ax.set_ylabel("Win rate (win=1, draw=0.5)")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.set_title("Alternating evaluation per round")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def summarize_series(series_rows: List[dict], out_path: Path) -> None:
    headers = ["phase", "name_a", "name_b", "games", "wins_a", "draws", "wins_b", "points_a", "points_b"]
    with out_path.open("w") as f:
        f.write("\t".join(headers) + "\n")
        for row in series_rows:
            f.write("\t".join(str(row.get(h, "")) for h in headers) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot tournament run results.")
    parser.add_argument("run_dir", type=Path, help="Path to a tournament_runs/<timestamp> directory")
    parser.add_argument("--eval-games-per-round", type=int, default=12, help="Games used in alternating_eval per round (matches training-eval-games)")
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    series_path = run_dir / "series_results.csv"
    games_path = run_dir / "game_results.csv"
    ranking_path = run_dir / "ranking.csv"
    training_path = run_dir / "training_rounds.csv"

    series_rows = load_csv(series_path) if series_path.exists() else []
    game_rows = load_csv(games_path) if games_path.exists() else []
    training_rows = load_csv(training_path) if training_path.exists() else []
    ranking = load_ranking(ranking_path) if ranking_path.exists() else []

    # Plots
    names, matrix = build_heatmap(series_rows, phases=["round_robin", "round_robin_minimax"])
    if names:
        plot_heatmap(names, matrix, run_dir / "heatmap_points.png", "Round-robin points (including minimax)")

    if ranking:
        plot_ranking_bar(ranking, run_dir / "ranking.png")

    rounds, winrates = compute_alternating_winrates(game_rows, args.eval_games_per_round)
    if rounds:
        plot_alternating_eval(rounds, winrates, run_dir / "alternating_eval.png")

    # Human-readable table dump for quick inspection
    summarize_series(series_rows, run_dir / "series_results.tsv")

    print(f"Generated plots in {run_dir}")
    if series_rows:
        print(f"- heatmap_points.png (round robin + minimax)" if names else "- skipped heatmap (no names)")
    if ranking:
        print("- ranking.png")
    if rounds:
        print("- alternating_eval.png")
    if training_rows:
        print("- training_rounds.csv present (not plotted directly)")


if __name__ == "__main__":
    main()
