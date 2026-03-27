import argparse
import random
import time
from collections import deque
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from ..checkers.board import Board
from ..constants import BLACK, WHITE, ROWS, COLS
from .algorithm import minimax, get_all_moves
from .rl_algorithm import (
    RLAgent,
    WIN_REWARD,
    LOSE_REWARD,
    DRAW_REWARD,
)
from .training import reset_board


class MinimaxOpponent:
    def __init__(self, depth: int) -> None:
        self.depth = depth
        self.name = f"minimax_d{depth}"

    def move(self, board: Board, color) -> Tuple[Board, str]:
        max_player = color == BLACK
        _, new_board, move, _, _ = minimax(deepcopy(board), self.depth, max_player, None)
        if move is None:
            return None, "no_moves"
        winner = new_board.winner()
        return new_board, winner.lower() if winner else None


class RLOpponent:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.name = path.name
        self.agent = RLAgent(model_path=path)
        self.agent.load()

    def move(self, board: Board, color) -> Tuple[Board, str]:
        _, _, entry = self.agent.select_entry(board, color, train=False)
        if entry is None:
            return None, "no_moves"
        next_board, _, _, _ = entry
        winner = next_board.winner()
        return next_board, winner.lower() if winner else None


def agent_step(agent: RLAgent, board: Board, color, train: bool, reward_on_loss: float = LOSE_REWARD):
    state, action_vec, entry = agent.select_entry(board, color, train=train)
    if entry is None:
        # no moves available
        base_state = state if state is not None else agent.encode_board(board, color)
        base_action = action_vec if action_vec is not None else agent.encode_action((0, 0, 0, 0))
        next_state = agent.encode_board(board, color)
        agent.train_short_memory(base_state, base_action, reward_on_loss, next_state, [], True)
        agent.remember(base_state, base_action, reward_on_loss, next_state, [], True)
        return None, True, "no_moves"

    next_board, move, piece, removed = entry
    shaped_reward = agent.compute_reward(next_board, removed, color)
    winner = next_board.winner()

    if winner:
        outcome = winner.lower()
        final_reward = WIN_REWARD if outcome == ("black" if color == BLACK else "white") else LOSE_REWARD
        next_state = agent.encode_board(next_board, color)
        agent.train_short_memory(state, action_vec, final_reward, next_state, [], True)
        agent.remember(state, action_vec, final_reward, next_state, [], True)
        return next_board, True, outcome

    return next_board, False, None


def play_episode_vs_opponent(agent: RLAgent, opponent, agent_color, max_turns: int) -> Tuple[str, str]:
    board = reset_board()
    current_color = BLACK
    turns = 0
    outcome = None

    while turns < max_turns:
        if current_color == agent_color:
            next_board, done, outcome = agent_step(agent, board, agent_color, train=True)
            if next_board is None:
                return ("white" if agent_color == BLACK else "black"), "agent_no_moves"
            board = next_board
            if done:
                return outcome, "agent_win" if outcome == ("black" if agent_color == BLACK else "white") else "agent_loss"
        else:
            next_board, opp_outcome = opponent.move(board, current_color)
            if next_board is None:
                return "black" if agent_color == BLACK else "white", "opp_no_moves"
            board = next_board
            if opp_outcome:
                return opp_outcome, "opp_win"

        current_color = WHITE if current_color == BLACK else BLACK
        turns += 1

    return "draw", "turn_cap"


def summarize_outcomes(outcomes: List[str]) -> Dict[str, int]:
    return {
        "black": outcomes.count("black"),
        "white": outcomes.count("white"),
        "draw": outcomes.count("draw"),
    }


def train_mixture(
    episodes: int,
    opponent_models: List[Path],
    minimax_depths: List[int],
    target_win_rate: float,
    window: int,
    max_turns: int,
    save_dir: Path,
    plot: bool,
    plot_every: int,
) -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / f"rl_mixture_{run_id}.pth"
    log_csv = save_dir / f"rl_mixture_{run_id}_log.csv"
    plot_path = save_dir / f"rl_mixture_{run_id}_winrate.png"

    agent = RLAgent(model_path=model_path)

    opponents = [MinimaxOpponent(d) for d in minimax_depths]
    opponents.extend(RLOpponent(p) for p in opponent_models)
    if not opponents:
        raise SystemExit("No opponents provided for mixture training.")

    recent = deque(maxlen=window)
    episodes_axis: List[int] = []
    win_rates: List[float] = []

    fig = None
    ax = None
    if plot:
        plt.ion()
        fig, ax = plt.subplots(figsize=(9, 4))

    with log_csv.open("w") as f:
        f.write("episode,opponent,agent_color,outcome,rolling_winrate\n")

    for ep in range(1, episodes + 1):
        opponent = random.choice(opponents)
        agent_color = BLACK if ep % 2 == 1 else WHITE
        outcome, reason = play_episode_vs_opponent(agent, opponent, agent_color, max_turns)

        score = 1.0 if (outcome == "black" and agent_color == BLACK) or (outcome == "white" and agent_color == WHITE) else 0.5 if outcome == "draw" else 0.0
        recent.append(score)
        win_rate = sum(recent) / len(recent)
        episodes_axis.append(ep)
        win_rates.append(win_rate)

        agent.n_games += 1
        agent.train_long_memory()
        agent.update_epsilon(win_rate, target_win_rate)

        with log_csv.open("a") as f:
            f.write(f"{ep},{opponent.name},{'black' if agent_color==BLACK else 'white'},{outcome},{win_rate:.4f}\n")

        if plot and ep % plot_every == 0:
            ax.cla()
            ax.plot(episodes_axis, win_rates, label=f"Rolling win rate (last {window})")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Win rate")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_title("Mixture training vs multiple opponents")
            plt.tight_layout()
            plt.pause(0.001)

        if ep % max(1, episodes // 20) == 0:
            print(f"Episode {ep}/{episodes}: opp={opponent.name}, outcome={outcome}, reason={reason}, rolling_winrate={win_rate:.3f}")

    agent.save()

    if plot and fig is not None and ax is not None:
        ax.plot(episodes_axis, win_rates, label=f"Rolling win rate (last {window})")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Win rate")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title("Mixture training vs multiple opponents")
        plt.tight_layout()
        fig.savefig(plot_path, dpi=200)
        plt.ioff()
        plt.close(fig)
    else:
        # fallback static plot
        fig2, ax2 = plt.subplots(figsize=(9, 4))
        ax2.plot(episodes_axis, win_rates, label=f"Rolling win rate (last {window})")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Win rate")
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_title("Mixture training vs multiple opponents")
        plt.tight_layout()
        fig2.savefig(plot_path, dpi=200)
        plt.close(fig2)
    print(f"Saved model to {model_path}\nSaved log to {log_csv}\nSaved plot to {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RL agent against a mixture of opponents (minimax + frozen RL models)")
    parser.add_argument("--episodes", type=int, default=1000, help="Training episodes")
    parser.add_argument("--opponent-models", nargs="*", type=Path, default=[], help="Paths to frozen opponent RL models")
    parser.add_argument("--minimax-depths", nargs="*", type=int, default=[1], help="List of minimax depths to include")
    parser.add_argument("--target-win-rate", type=float, default=0.8, help="Win-rate target used for epsilon schedule")
    parser.add_argument("--window", type=int, default=100, help="Rolling window for win rate")
    parser.add_argument("--max-turns", type=int, default=150, help="Max plies per game to reduce turn-cap draws")
    parser.add_argument("--save-dir", type=Path, default=Path(__file__).resolve().parent.parent / "model", help="Directory to save models and logs")
    parser.add_argument("--no-plot", action="store_true", help="Disable live plotting (still saves final PNG)")
    parser.add_argument("--plot-every", type=int, default=10, help="Update live plot every N episodes")
    args = parser.parse_args()

    train_mixture(
        episodes=args.episodes,
        opponent_models=args.opponent_models,
        minimax_depths=args.minimax_depths,
        target_win_rate=args.target_win_rate,
        window=args.window,
        max_turns=args.max_turns,
        save_dir=args.save_dir,
        plot=not args.no_plot,
        plot_every=args.plot_every,
    )


if __name__ == "__main__":
    main()
