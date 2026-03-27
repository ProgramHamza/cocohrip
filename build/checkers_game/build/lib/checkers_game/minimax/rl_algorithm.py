import math
import random
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from ..constants import BLACK, WHITE
from .algorithm import get_all_moves
from .rl_model import DQNTrainer, StateActionQNet

WIN_REWARD = 100.0
LOSE_REWARD = -100.0
DRAW_REWARD = -50.0
CAPTURE_REWARD = 8.0
MATERIAL_WEIGHT = 0.05

MAX_MEMORY = 50_000
BATCH_SIZE = 256
LR = 0.001
STATE_SIZE = 65  # turn flag + 64 board squares
ACTION_SIZE = 4  # from_row, from_col, to_row, to_col
TRAIN_EVERY = 1  # train every step
TARGET_UPDATE_EVERY = 500  # steps
POLYAK_TAU = 0.05


class RLAgent:
    """DQN-style agent for checkers using (state, action) pairs."""

    def __init__(
        self,
        gamma: float = 0.95,
        epsilon_start: float = 0.3,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.002,
        model_path: str | Path | None = None,
    ) -> None:
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_games = 0
        self.total_steps = 0
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = StateActionQNet(STATE_SIZE, ACTION_SIZE)
        self.target_model = StateActionQNet(STATE_SIZE, ACTION_SIZE)
        self.target_model.load_state_dict(self.model.state_dict())
        optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9995)
        self.trainer = DQNTrainer(self.model, lr=LR, gamma=gamma, optimizer=optimizer, scheduler=scheduler)
        self.model_path = Path(model_path) if model_path else (Path(__file__).resolve().parent.parent / "model" / "rl_model.pth")
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.train_step_counter = 0
        # Simple epsilon schedule (per-game decay)
        self.epsilon_value = self.epsilon_start
        self.epsilon_low = self.epsilon_min
        self.epsilon_high = 0.6

    def save(self) -> None:
        torch.save(self.model.state_dict(), self.model_path)

    def load(self) -> None:
        if self.model_path.exists():
            self.model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
            self.model.eval()
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_model.eval()

    def _maybe_update_target(self) -> None:
        if self.train_step_counter % TARGET_UPDATE_EVERY == 0:
            # Polyak averaging for stability
            with torch.no_grad():
                for tp, p in zip(self.target_model.parameters(), self.model.parameters()):
                    tp.data.mul_(1 - POLYAK_TAU).add_(p.data * POLYAK_TAU)

    def epsilon(self) -> float:
        return max(self.epsilon_min, self.epsilon_value)

    def update_epsilon(self, current_winrate: float, target_winrate: float = 0.75) -> None:
        # per-episode linear decay toward epsilon_min, ignoring PID
        self.epsilon_value = max(self.epsilon_min, self.epsilon_start - self.epsilon_decay * self.n_games)

    def encode_board(self, board, color: Tuple[int, int, int]) -> np.ndarray:
        grid: List[int] = []
        for row in board.board:
            for cell in row:
                if cell == 0:
                    grid.append(0)
                else:
                    if cell.color == color:
                        grid.append(3 if cell.king else 1)
                    else:
                        grid.append(4 if cell.king else 2)
        return np.array([1 if color == BLACK else 0, *grid], dtype=float)

    def action_key(self, piece, move: Tuple[int, int]) -> Tuple[int, int, int, int]:
        return (piece.row, piece.col, move[0], move[1])

    def encode_action(self, action: Tuple[int, int, int, int]) -> np.ndarray:
        norm = 7.0
        return np.array([action[0] / norm, action[1] / norm, action[2] / norm, action[3] / norm], dtype=float)

    def material_score(self, board, color: Tuple[int, int, int]) -> float:
        score = board.evaluate()
        return score if color == BLACK else -score

    def compute_reward(self, board_after, removed, color: Tuple[int, int, int]) -> float:
        winner = board_after.winner()
        if winner is not None:
            if winner == "BLACK":
                return WIN_REWARD if color == BLACK else LOSE_REWARD
            if winner == "WHITE":
                return WIN_REWARD if color == WHITE else LOSE_REWARD
            return DRAW_REWARD
        if removed:
            return CAPTURE_REWARD
        shaped = MATERIAL_WEIGHT * self.material_score(board_after, color)
        reward = shaped
        return max(-1.0, min(1.0, reward))

    def available_action_vectors(self, board, color) -> List[np.ndarray]:
        entries = get_all_moves(board, color, None)
        return [self.encode_action(self.action_key(piece, move)) for _, move, piece, _ in entries] if entries else []

    def select_entry(self, board, color, train: bool):
        candidates = get_all_moves(board, color, None)
        if not candidates:
            return None, None, None

        state_vec = self.encode_board(board, color)
        if train and random.random() < self.epsilon():
            next_board, move, piece, removed = random.choice(candidates)
            action_vec = self.encode_action(self.action_key(piece, move))
            return state_vec, action_vec, (next_board, move, piece, removed)

        state_tensor = torch.tensor(state_vec, dtype=torch.float).unsqueeze(0)
        best_q = -math.inf
        best_entry = None
        best_action_vec = None
        for entry in candidates:
            next_board, move, piece, removed = entry
            action_vec = self.encode_action(self.action_key(piece, move))
            action_tensor = torch.tensor(action_vec, dtype=torch.float).unsqueeze(0)
            q_val = self.model(state_tensor, action_tensor).item()
            if q_val > best_q:
                best_q = q_val
                best_entry = entry
                best_action_vec = action_vec

        return state_vec, best_action_vec, best_entry

    def remember(self, state, action, reward, next_state, next_actions, done) -> None:
        self.memory.append((state, action, reward, next_state, next_actions, done))

    def train_long_memory(self) -> None:
        if len(self.memory) == 0:
            return
        batch = random.sample(self.memory, min(len(self.memory), BATCH_SIZE))
        states, actions, rewards, next_states, next_actions, dones = zip(*batch)
        self.trainer.train_step(states, actions, rewards, next_states, dones, next_actions, self.target_model)
        self.train_step_counter += 1
        self.total_steps += 1
        self._maybe_update_target()

    def train_short_memory(self, state, action, reward, next_state, next_actions, done) -> None:
        self.train_step_counter += 1
        self.total_steps += 1
        if self.train_step_counter % TRAIN_EVERY != 0:
            return
        self.trainer.train_step([state], [action], [reward], [next_state], [done], [next_actions], self.target_model)
        self._maybe_update_target()


def rl_move(
    board,
    color: Tuple[int, int, int] = BLACK,
    agent: RLAgent = None,
    train: bool = False,
) -> Tuple[float, object, Tuple[int, int], object, List[object]]:
    """Return the best move using the RL agent (DQN policy)."""
    rl_agent = agent or RLAgent()
    state, action_vec, entry = rl_agent.select_entry(board, color, train=train)
    if entry is None:
        return float("-inf"), board, None, None, None

    next_board, move, piece, removed = entry
    reward = rl_agent.compute_reward(next_board, removed, color)

    if train:
        next_state = rl_agent.encode_board(next_board, color)
        next_actions = rl_agent.available_action_vectors(next_board, color)
        rl_agent.train_short_memory(state, action_vec, reward, next_state, next_actions, False)

    return reward, next_board, move, piece, removed
