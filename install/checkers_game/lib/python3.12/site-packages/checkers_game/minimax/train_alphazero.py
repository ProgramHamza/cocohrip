import argparse
import random
import time
from collections import deque
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import torch
import matplotlib.pyplot as plt

from ..constants import BLACK, WHITE
from .alphazero_model import AZNet, AZTrainer, action_index, ACTION_SIZE
from .alphazero_mcts import encode_board_planes, mcts_search, legal_action_indices, action_index_to_coords
from .algorithm import get_all_moves, minimax
from .training import reset_board


def play_self_play_game(model: AZNet, sims: int, max_turns: int, temperature: float = 1.0):
    states: List[torch.Tensor] = []
    policies: List[torch.Tensor] = []
    players: List[int] = []

    board = reset_board()
    to_play = BLACK
    turns = 0
    winner = None

    while turns < max_turns:
        pi = mcts_search(board, to_play, model, sims=sims)
        policies.append(pi)
        states.append(encode_board_planes(board, to_play))
        players.append(1 if to_play == BLACK else -1)

        if temperature <= 1e-6:
            action_idx = torch.argmax(pi).item()
        else:
            probs = pi ** (1.0 / temperature)
            probs = probs / probs.sum()
            action_idx = torch.multinomial(probs, 1).item()

        moves = get_all_moves(board, to_play, None)
        chosen_board = None
        for nb, move, piece, _ in moves:
            if action_index(piece.row, piece.col, move[0], move[1]) == action_idx:
                chosen_board = nb
                break
        if chosen_board is None:
            # illegal (masked out) -> pick any legal move
            if not moves:
                winner = "WHITE" if to_play == BLACK else "BLACK"
                break
            chosen_board = random.choice(moves)[0]

        board = chosen_board
        winner = board.winner()
        if winner:
            break
        to_play = WHITE if to_play == BLACK else BLACK
        turns += 1

    if winner is None:
        z_final = 0.0
    else:
        z_final = 1.0 if winner == "BLACK" else -1.0

    targets = []
    for p in players:
        targets.append(z_final * p)
    return states, policies, targets, winner or "DRAW"


def evaluate_vs_minimax(model: AZNet, depth: int, games: int, sims: int, max_turns: int) -> float:
    wins = 0.0
    for g in range(games):
        board = reset_board()
        to_play = BLACK if g % 2 == 0 else WHITE
        turns = 0
        winner = None
        while turns < max_turns:
            if to_play == BLACK:
                pi = mcts_search(board, BLACK, model, sims=sims)
                action_idx = torch.argmax(pi).item()
                moves = get_all_moves(board, BLACK, None)
                chosen_board = None
                for nb, move, piece, _ in moves:
                    if action_index(piece.row, piece.col, move[0], move[1]) == action_idx:
                        chosen_board = nb
                        break
                if chosen_board is None:
                    if not moves:
                        winner = "WHITE"
                        break
                    chosen_board = random.choice(moves)[0]
                board = chosen_board
            else:
                _, nb, move, _, _ = minimax(deepcopy(board), depth, to_play == BLACK, None)
                if move is None:
                    winner = "BLACK"
                    break
                board = nb
            winner = board.winner()
            if winner:
                break
            to_play = WHITE if to_play == BLACK else BLACK
            turns += 1
        if winner is None:
            wins += 0.5
        elif winner == "BLACK":
            wins += 1.0 if g % 2 == 0 else 0.0
        elif winner == "WHITE":
            wins += 1.0 if g % 2 == 1 else 0.0
    return wins / games


def train_alphazero(
    episodes: int,
    sims: int,
    max_turns: int,
    buffer_size: int,
    batch_size: int,
    eval_every: int,
    eval_games: int,
    minimax_depth: int,
    save_dir: Path,
) -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / f"az_model_{run_id}.pth"
    plot_path = save_dir / f"az_model_{run_id}_winrate.png"

    model = AZNet()
    trainer = AZTrainer(model, lr=1e-3)

    buffer_states: deque = deque(maxlen=buffer_size)
    buffer_policies: deque = deque(maxlen=buffer_size)
    buffer_values: deque = deque(maxlen=buffer_size)

    winrates: List[float] = []
    eval_axis: List[int] = []

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 4))

    for ep in range(1, episodes + 1):
        states, policies, targets, outcome = play_self_play_game(model, sims=sims, max_turns=max_turns, temperature=1.0 if ep < 30 else 0.1)
        buffer_states.extend(states)
        buffer_policies.extend(policies)
        buffer_values.extend(targets)

        if len(buffer_states) >= batch_size:
            idxs = random.sample(range(len(buffer_states)), batch_size)
            batch_states = torch.stack([buffer_states[i] for i in idxs])
            batch_policies = torch.stack([buffer_policies[i] for i in idxs])
            batch_values = torch.tensor([buffer_values[i] for i in idxs], dtype=torch.float32)
            trainer.train_step(batch_states, batch_policies, batch_values)

        if ep % eval_every == 0:
            winrate = evaluate_vs_minimax(model, depth=minimax_depth, games=eval_games, sims=max(8, sims // 2), max_turns=max_turns)
            winrates.append(winrate)
            eval_axis.append(ep)
            ax.cla()
            ax.plot(eval_axis, winrates, marker="o")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Win rate vs minimax")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.set_title("AlphaZero-lite eval vs minimax")
            plt.tight_layout()
            plt.pause(0.001)
            print(f"Eval @ episode {ep}: winrate={winrate:.3f}")

    torch.save(model.state_dict(), model_path)
    plt.ioff()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved model {model_path}\nSaved eval plot {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="AlphaZero-lite self-play training for checkers")
    parser.add_argument("--episodes", type=int, default=200, help="Self-play games")
    parser.add_argument("--sims", type=int, default=32, help="MCTS simulations per move")
    parser.add_argument("--max-turns", type=int, default=150, help="Turn cap per game")
    parser.add_argument("--buffer-size", type=int, default=20000, help="Replay buffer for (state,pi,z)")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size")
    parser.add_argument("--eval-every", type=int, default=20, help="Episodes between evals vs minimax")
    parser.add_argument("--eval-games", type=int, default=20, help="Games for each eval vs minimax")
    parser.add_argument("--minimax-depth", type=int, default=1, help="Depth for minimax eval opponent")
    parser.add_argument("--save-dir", type=Path, default=Path(__file__).resolve().parent.parent / "model", help="Directory to save models/plots")
    args = parser.parse_args()

    train_alphazero(
        episodes=args.episodes,
        sims=args.sims,
        max_turns=args.max_turns,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        minimax_depth=args.minimax_depth,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
