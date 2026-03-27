import math
import random
from copy import deepcopy
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from ..constants import BLACK, WHITE
from ..checkers.board import Board
from .algorithm import get_all_moves
from .alphazero_model import action_index, AZNet, ACTION_SIZE


class AZNode:
    def __init__(self, board: Board, to_play, parent=None, prior: float = 0.0) -> None:
        self.board = board
        self.to_play = to_play
        self.parent = parent
        self.prior = prior
        self.children: Dict[int, "AZNode"] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
        self.legal_actions: List[int] = []

    def value(self) -> float:
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count

    def expand(self, policy_logits: torch.Tensor, legal_actions: List[int]):
        self.is_expanded = True
        self.legal_actions = legal_actions
        if not legal_actions:
            return
        with torch.no_grad():
            logits = policy_logits[legal_actions]
            probs = F.softmax(logits, dim=0).cpu().numpy()
        for a, p in zip(legal_actions, probs):
            self.children[a] = AZNode(board=None, to_play=None, parent=self, prior=float(p))

    def select_child(self, c_puct: float):
        best_score = -1e9
        best_action = None
        best_child = None
        for action, child in self.children.items():
            pb_c = c_puct * child.prior * math.sqrt(self.visit_count + 1) / (child.visit_count + 1)
            score = child.value() + pb_c
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child


def encode_board_planes(board: Board, to_play) -> torch.Tensor:
    planes = []
    for color in (BLACK, WHITE):
        men = [[0.0] * 8 for _ in range(8)]
        kings = [[0.0] * 8 for _ in range(8)]
        for r, row in enumerate(board.board):
            for c, piece in enumerate(row):
                if piece == 0:
                    continue
                if piece.color != color:
                    continue
                if piece.king:
                    kings[r][c] = 1.0
                else:
                    men[r][c] = 1.0
        planes.append(men)
        planes.append(kings)
    stm_plane = [[1.0 if to_play == BLACK else 0.0] * 8 for _ in range(8)]
    planes.append(stm_plane)
    return torch.tensor(planes, dtype=torch.float32)


def legal_action_indices(board: Board, color) -> List[int]:
    moves = get_all_moves(board, color, None)
    indices: List[int] = []
    for nb, move, piece, _ in moves:
        fr, fc = piece.row, piece.col
        tr, tc = move
        indices.append(action_index(fr, fc, tr, tc))
    return indices


def step_board(board: Board, action_idx: int, color):
    fr, fc, tr, tc = action_index_to_coords(action_idx)
    # rebuild move through get_all_moves to ensure legality and capture info
    moves = get_all_moves(board, color, None)
    for nb, move, piece, _ in moves:
        if piece.row == fr and piece.col == fc and move[0] == tr and move[1] == tc:
            return nb
    return None


def action_index_to_coords(idx: int):
    from .alphazero_model import index_to_action
    return index_to_action(idx)


def mcts_search(root_board: Board, to_play, model: AZNet, sims: int = 32, c_puct: float = 1.5):
    root = AZNode(board=root_board, to_play=to_play, parent=None, prior=1.0)

    for _ in range(sims):
        node = root
        search_path = [node]

        # Selection
        while node.is_expanded and node.children:
            action, node = node.select_child(c_puct)
            if node.board is None:
                parent_board = search_path[-1].board
                node_board = step_board(parent_board, action, search_path[-1].to_play)
                node.board = node_board
                node.to_play = WHITE if search_path[-1].to_play == BLACK else BLACK
            search_path.append(node)
            if node.board is None:
                break
            winner = node.board.winner()
            if winner is not None:
                break

        leaf = search_path[-1]
        if leaf.board is None:
            value = 0.0
        else:
            winner = leaf.board.winner()
            if winner == "BLACK":
                value = 1.0 if leaf.to_play == WHITE else -1.0
            elif winner == "WHITE":
                value = 1.0 if leaf.to_play == BLACK else -1.0
            else:
                # Expand
                state_planes = encode_board_planes(leaf.board, leaf.to_play).unsqueeze(0)
                with torch.no_grad():
                    policy_logits, value_pred = model(state_planes)
                legal = legal_action_indices(leaf.board, leaf.to_play)
                leaf.expand(policy_logits.squeeze(0), legal)
                value = float(value_pred.item())

        # Backpropagate
        for n in reversed(search_path):
            n.visit_count += 1
            n.value_sum += value if n.to_play == to_play else -value

    # Build policy target from visit counts
    pi = torch.zeros((ACTION_SIZE,), dtype=torch.float32)
    total = sum(child.visit_count for child in root.children.values())
    if total > 0:
        for action, child in root.children.items():
            pi[action] = child.visit_count / total
    else:
        legal = legal_action_indices(root_board, to_play)
        for a in legal:
            pi[a] = 1.0 / len(legal) if legal else 0.0
    return pi
