import argparse
import random
import time
from collections import deque
from copy import deepcopy
from typing import List
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

from ..checkers.board import Board
from ..constants import BLACK, WHITE, ROWS, COLS
from .algorithm import get_all_moves, minimax
from .rl_algorithm import (
    DRAW_REWARD,
    LOSE_REWARD,
    WIN_REWARD,
    RLAgent,
)


def initial_board_matrix() -> List[List[int]]:
    layout: List[List[int]] = []
    for row in range(ROWS):
        row_values: List[int] = []
        for col in range(COLS):
            if (row + col) % 2 == 0:
                row_values.append(0)
            elif row < 3:
                row_values.append(2)  # black man in camera encoding
            elif row > 4:
                row_values.append(1)  # white man in camera encoding
            else:
                row_values.append(0)
        layout.append(row_values)
    return layout


def reset_board() -> Board:
    board = Board()
    board.create_board(initial_board_matrix())
    return board


def random_move(board: Board, color) -> List:
    moves = get_all_moves(board, color, None)
    if not moves:
        return []
    return random.choice(moves)


def opponent_reply(board: Board, opponent_depth: int) -> (Board, str):
    if opponent_depth <= 0:
        choice = random_move(board, WHITE)
        if not choice:
            return board, "black"
        return choice[0], None
    _, new_board, move, _, _ = minimax(deepcopy(board), opponent_depth, False, None)
    if move is None:
        return board, "black"
    winner = new_board.winner()
    if winner:
        return new_board, winner.lower()
    return new_board, None


def play_episode(agent: RLAgent, opponent_depth: int, max_turns: int = 200, train: bool = True) -> str:
    board = reset_board()
    turns = 0
    outcome = None

    while turns < max_turns:
        state, action_vec, entry = agent.select_entry(board, BLACK, train=train)
        if entry is None:
            outcome = "white"
            break

        next_board, move, piece, removed = entry
        reward = agent.compute_reward(next_board, removed, BLACK)
        board_after_agent = next_board
        board_after_opp = board_after_agent

        winner = board_after_agent.winner()
        if winner:
            outcome = winner.lower()
            done = True
            next_state = agent.encode_board(board_after_agent, BLACK)
            next_actions = []
        else:
            board_after_opp, opp_outcome = opponent_reply(board_after_agent, opponent_depth)
            outcome = opp_outcome
            done = outcome is not None
            next_state = agent.encode_board(board_after_opp, BLACK)
            next_actions = agent.available_action_vectors(board_after_opp, BLACK) if not done else []

        final_reward = reward
        if outcome == "white":
            final_reward = LOSE_REWARD
        elif outcome == "black":
            final_reward = WIN_REWARD
        elif outcome == "draw":
            final_reward = DRAW_REWARD

        if train:
            agent.train_short_memory(state, action_vec, final_reward, next_state, next_actions, done)
            agent.remember(state, action_vec, final_reward, next_state, next_actions, done)

        board = board_after_agent if winner else board_after_opp

        if done:
            if train:
                agent.n_games += 1
                agent.train_long_memory()
            return outcome

        turns += 1

    if train:
        agent.n_games += 1
        agent.train_long_memory()
    return outcome or "draw"


def train(
    episodes: int,
    opponent_depth: int,
    target_win_rate: float = 0.8,
    window: int = 100,
    max_turns: int = 120,
    plot: bool = True,
    plot_every: int = 10,
) -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = (Path(__file__).resolve().parent.parent / "model")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"rl_model_{run_id}.pth"

    agent = RLAgent(model_path=model_path)
    recent = deque(maxlen=window)
    win_rates: List[float] = []
    elapsed_minutes: List[float] = []
    episodes_axis: List[int] = []
    best_win = 0.0

    t0 = time.time()
    plt.ion()
    fig = plt.figure(figsize=(12, 4)) if plot else None

    ep = 0
    while True:
        ep += 1
        # Add a bit of opponent stochasticity to reduce overfitting
        current_depth = max(0, opponent_depth + random.choice([-1, 0, 1]))
        outcome = play_episode(agent, current_depth, max_turns=max_turns, train=True)
        # encode outcome for win rate: win=1, draw=0.5, loss=0
        recent.append(1.0 if outcome == "black" else 0.5 if outcome == "draw" else 0.0)

        win_rate = sum(recent) / len(recent)
        episodes_axis.append(ep)
        win_rates.append(win_rate)
        elapsed_minutes.append((time.time() - t0) / 60.0)

        # PID-based adaptive exploration (once per episode)
        agent.update_epsilon(win_rate, target_win_rate)

        if ep % max(1, episodes // 10) == 0 or len(recent) == window:
            print(
                f"Episode {ep}: outcome={outcome}, "
                f"rolling_win_rate={win_rate:.3f} over last {len(recent)}"
            )

        # live plot update
        if plot and (ep % plot_every == 0):
            plt.clf()
            ax1 = plt.subplot(1, 2, 1)
            ax1.plot(episodes_axis, win_rates, label="Rolling win rate")
            ax1.set_xlabel("Episode")
            ax1.set_ylabel("Win rate (last 100)")
            ax1.set_ylim(0, 1)
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            ax2 = plt.subplot(1, 2, 2)
            ax2.plot(elapsed_minutes, win_rates, label="Win rate vs time")
            ax2.set_xlabel("Elapsed minutes")
            ax2.set_ylabel("Win rate")
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            fig.suptitle("RL vs minimax (rolling win rate)")
            plt.tight_layout()
            plt.pause(0.001)

        # save on improvement and on termination
        if win_rate > best_win and len(recent) >= 10:
            best_win = win_rate
            agent.save()

        # stop when target met with full window
        if len(recent) == window and win_rate >= target_win_rate:
            print(f"Target win rate {win_rate:.3f} reached at episode {ep}. Freezing and evaluating...")
            agent.save()

            eval_results = []
            for _ in range(100):
                eval_outcome = play_episode(agent, opponent_depth, max_turns=max_turns, train=False)
                eval_results.append(1.0 if eval_outcome == "black" else 0.5 if eval_outcome == "draw" else 0.0)
            eval_win = sum(eval_results) / len(eval_results)
            print(f"Evaluation win rate over 100 games: {eval_win:.3f}")

            if eval_win >= target_win_rate:
                print(f"Model accepted and saved at {model_path}")
                agent.save()
            else:
                print("Evaluation below target; model not promoted. Continuing training...")
                # reset recent to avoid immediate stop
                recent.clear()
                win_rates.append(eval_win)
                episodes_axis.append(ep)
                elapsed_minutes.append((time.time() - t0) / 60.0)
            # continue training even after eval if not accepted
            if eval_win >= target_win_rate:
                break

        # safety cap
        if ep >= episodes:
            print(f"Reached episode cap {episodes} with win rate {win_rate:.3f}. Saving model.")
            agent.save()
            break

    if plot:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL checkers agent with quick self-play")
    parser.add_argument("--episodes", type=int, default=1000, help="Maximum self-play episodes (cap)")
    parser.add_argument("--opponent-depth", type=int, default=1, help="Minimax depth for opponent (0=random)")
    parser.add_argument("--target-win-rate", type=float, default=0.8, help="Rolling win rate target (last 100 games)")
    parser.add_argument("--max-turns", type=int, default=120, help="Max ply per episode (lower for faster runs)")
    parser.add_argument("--no-plot", action="store_true", help="Disable live plotting for speed")
    args = parser.parse_args()

    train(
        episodes=args.episodes,
        opponent_depth=args.opponent_depth,
        target_win_rate=args.target_win_rate,
        max_turns=args.max_turns,
        plot=not args.no_plot,
    )
