import argparse
import random
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from ..checkers.board import Board
from ..constants import BLACK, WHITE, ROWS, COLS
from .algorithm import minimax, get_all_moves
from .rl_algorithm import RLAgent


def initial_board_matrix() -> List[List[int]]:
    layout: List[List[int]] = []
    for row in range(ROWS):
        row_values: List[int] = []
        for col in range(COLS):
            if (row + col) % 2 == 0:
                row_values.append(0)
            elif row < 3:
                row_values.append(2)
            elif row > 4:
                row_values.append(1)
            else:
                row_values.append(0)
        layout.append(row_values)
    return layout


def reset_board() -> Board:
    board = Board()
    board.create_board(initial_board_matrix())
    return board


def opponent_reply(board: Board, depth: int) -> Board:
    if depth <= 0:
        moves = get_all_moves(board, WHITE, None)
        if not moves:
            return board
        next_board, _, _, _ = random.choice(moves)
        return next_board
    _, new_board, move, _, _ = minimax(deepcopy(board), depth, False, None)
    return new_board if move is not None else board


def play_game(agent: RLAgent, minimax_depth: int, max_turns: int) -> str:
    board = reset_board()
    current_color = BLACK  # agent as black
    turns = 0

    while turns < max_turns:
        if current_color == BLACK:
            state, _, entry = agent.select_entry(board, BLACK, train=False)
            if entry is None:
                return "white"
            board, _, _, _ = entry
            winner = board.winner()
            if winner:
                return winner.lower()
            current_color = WHITE
        else:
            board = opponent_reply(board, minimax_depth)
            winner = board.winner()
            if winner:
                return winner.lower()
            current_color = BLACK
        turns += 1

    return "draw"


def roll_winrate(outcomes: deque) -> float:
    if not outcomes:
        return 0.0
    return sum(outcomes) / len(outcomes)


def evaluate(agent_path: Path, games: int, minimax_depth: int, window: int, max_turns: int, out_path: Path) -> None:
    agent = RLAgent(model_path=agent_path)
    agent.load()

    window_scores: deque = deque(maxlen=window)
    xs: List[int] = []
    ys: List[float] = []

    outcomes: List[str] = []
    draws_by_turn_cap = 0

    for g in range(1, games + 1):
        outcome = play_game(agent, minimax_depth, max_turns)
        outcomes.append(outcome)
        if outcome == "black":
            window_scores.append(1.0)
        elif outcome == "draw":
            window_scores.append(0.5)
            draws_by_turn_cap += 1
        else:
            window_scores.append(0.0)

        xs.append(g)
        ys.append(roll_winrate(window_scores))

        if g % max(1, games // 10) == 0:
            print(f"Game {g}/{games}: outcome={outcome}, rolling_winrate_{window}={ys[-1]:.3f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, ys, label=f"Rolling win rate (last {window})")
    ax.set_xlabel("Game")
    ax.set_ylabel("Win rate")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f"{agent_path.name} vs minimax depth {minimax_depth}")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {out_path}")
    print(
        f"Outcomes over {games} games: wins={outcomes.count('black')}, draws={outcomes.count('draw')}, "
        f"losses={outcomes.count('white')}, turn-cap draws (likely)={draws_by_turn_cap}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved RL model vs minimax and plot rolling win rate.")
    parser.add_argument("model_path", type=Path, help="Path to .pth model file")
    parser.add_argument("--games", type=int, default=1000, help="Number of games to play")
    parser.add_argument("--minimax-depth", type=int, default=1, help="Depth for minimax opponent")
    parser.add_argument("--window", type=int, default=50, help="Rolling window size for win rate")
    parser.add_argument("--max-turns", type=int, default=300, help="Max plies per game (increase to reduce turn-cap draws)")
    parser.add_argument("--out", type=Path, help="Output PNG path (default: alongside model)")
    args = parser.parse_args()

    out_path = args.out
    if out_path is None:
        out_path = args.model_path.with_name(args.model_path.stem + "_minimax_eval.png")

    evaluate(
        agent_path=args.model_path,
        games=args.games,
        minimax_depth=args.minimax_depth,
        window=args.window,
        max_turns=args.max_turns,
        out_path=out_path,
    )


if __name__ == "__main__":
    main()
