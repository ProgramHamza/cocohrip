from copy import deepcopy
import pygame
from ..constants import BLACK, WHITE


def minimax(board, depth, max_player, game):
    if depth == 0 or board.winner() != None:
        return board.evaluate(), board, None, None, None
    
    if max_player:
        maxEval = float('-inf')
        best_move = None
        best_piece = None
        best_board = board
        best_removed = []
        for new_board, move, piece, removed in get_all_moves(board, BLACK, game):
            evaluation = minimax(new_board, depth-1, False, game)[0]
            maxEval = max(maxEval, evaluation)
            if maxEval == evaluation:
                best_move = move
                best_piece = piece
                best_board = new_board
                best_removed = removed

        # if no moves were available, return current board state
        return maxEval, best_board, best_move, best_piece, best_removed
    else:
        minEval = float('inf')
        best_move = None
        best_piece = None
        best_board = board
        best_removed = []
        for new_board, move, piece, removed in get_all_moves(board, WHITE, game):
            evaluation = minimax(new_board, depth-1, True, game)[0]
            minEval = min(minEval, evaluation)
            if minEval == evaluation:
                best_move = move
                best_piece = piece
                best_board = new_board
                best_removed = removed

        return minEval, best_board, best_move, best_piece, best_removed

    
def simulate_move(piece, move, board, game, skip):
    board.move(piece, move[0], move[1])
    removed_pieces = []
    if skip:
        for skip_piece in skip:
            board.remove(skip_piece)
            removed_pieces.append(skip_piece)
    
    return board, removed_pieces

def get_all_moves(board, color, game):
    moves = []

    for piece in board.get_all_pieces_by_color(color):
        new_piece = deepcopy(piece)
        valid_moves = board.get_valid_moves(piece)
        for move, skip in valid_moves.items():
            temp_board = deepcopy(board)
            temp_piece = temp_board.get_piece(piece.row, piece.col)
            new_board, removed_pieces = simulate_move(temp_piece, move, temp_board, game, skip)
            moves.append([new_board, move, new_piece, removed_pieces])

    return moves

