from math import pi
from re import S
import re
from tkinter import NO, W
from matplotlib.pyplot import pie
import pygame
from ..constants import BLACK, ROWS, SQUARE_SIZE, COLS, WHITE, GREY, BROWN, BLUE
from .piece import Piece

class Board:
    def __init__(self):
        self.isBoardCreated = False
        self.board = []
        self.black_left = self.white_left = 12
        self.black_kings = self.white_kings = 0
        self.wasBoardChanged = False

    def draw_squares(self, win):
        win.fill(BROWN)
        for row in range(ROWS):
            for col in range(row % 2, ROWS, 2):
                pygame.draw.rect(win, GREY, (row*SQUARE_SIZE, col*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))


    def get_piece(self, row, col):
        return self.board[row][col]

    def create_board(self, boardDetected):
        self.board = []
        self.black_left = self.white_left = 0
        for row in range(ROWS):
            self.board.append([])
            for col in range(COLS):
                piece = boardDetected[row][col]
                if piece == 1:
                    self.board[row].append(Piece(row, col, WHITE))
                    self.white_left += 1
                elif piece == 2:
                    self.board[row].append(Piece(row, col, BLACK))
                    self.black_left += 1
                else:
                    self.board[row].append(0)
        
        if self.white_left == 12 and self.black_left == 12:
            self.isBoardCreated = True
            print(f"\n✓ Initial board created successfully: {self.black_left} black pieces, {self.white_left} white pieces\n")
        else:
            print(f"⚠ Board detection issue: black={self.black_left}, white={self.white_left} (expected 12 each)")
            self.black_left = self.white_left = 12

    def update_board(self, boardDetected):
        piece = None
        new_piece_row = None
        new_piece_col = None
        color_of_piece = None
        for row in range(ROWS):
            for col in range(COLS):
                if boardDetected[row][col] != 0 and self.board[row][col] == 0:
                    if(boardDetected[row][col] == 1):
                        color_of_piece = WHITE
                    else:
                        color_of_piece = BLACK

                    new_piece_row = row
                    new_piece_col = col
                    break

        for row in range(ROWS):
            for col in range(COLS):
                if boardDetected[row][col] == 0 and self.board[row][col] != 0 and self.board[row][col].color == color_of_piece:
                    piece = self.board[row][col]
                    break

        if piece != None and new_piece_row != None:
            self.board[piece.row][piece.col], self.board[new_piece_row][new_piece_col] = self.board[new_piece_row][new_piece_col], self.board[piece.row][piece.col]
            piece.move(new_piece_row, new_piece_col)

            self.check_king_field(piece, new_piece_row)
            self.wasBoardChanged = True

        for row in range(ROWS):
            for col in range(COLS):
                if boardDetected[row][col] == 0 and self.board[row][col] != 0:
                    piece = self.board[row][col]
                    self.remove(piece)

    def remove(self, piece):
        self.board[piece.row][piece.col] = 0
        if piece != 0:
            if piece.color == BLACK:
                self.black_left -= 1
            else:
                self.white_left -= 1

    def check_king_field(self, piece, row):
        if piece.color == WHITE:
            if row == 0:
                if piece.king == False:
                    piece.make_king()
                    self.white_kings += 1
        if piece.color == BLACK:
            if row == ROWS - 1:
                if piece.king == False:
                    piece.make_king()
                    self.black_kings += 1

    def draw(self, win, boardDetected):
        if self.isBoardCreated == False:
            self.create_board(boardDetected)
        else:
            self.update_board(boardDetected)

        self.draw_squares(win)

        if self.isBoardCreated == True:
            for row in range(ROWS):
                for col in range(COLS):
                    piece = self.board[row][col]
                    if piece != 0:
                        piece.draw(win)


    def winner(self):
        if self.white_left <= 0:
            return "BLACK"
        elif self.black_left <= 0:
            return "WHITE"
        
        return None

    def evaluate(self):
        return self.black_left - self.white_left + (self.black_left * 0.5 - self.white_kings * 0.5)
    
    def get_all_pieces_by_color(self, color):
        pieces = []
        for row in self.board:
            for piece in row:
                if piece != 0 and piece.color == color:
                    pieces.append(piece)
        return pieces
    
    def get_all_pieces(self):
        pieces = []
        for row in self.board:
            for piece in row:
                if piece != 0:
                    pieces.append(piece)
        return pieces
    
    def move(self, piece, row, col):
        self.board[piece.row][piece.col], self.board[row][col] = self.board[row][col], self.board[piece.row][piece.col]
        piece.move(row, col)

        if row == ROWS - 1 or row == 0:
            if not piece.king:
                piece.make_king()
                if piece.color == WHITE:
                    self.white_kings += 1
                else:
                    self.black_kings += 1
    
    def get_valid_moves(self, piece):
        moves = {}
        left = piece.col - 1
        right = piece.col + 1
        row = piece.row

        if piece.color == WHITE or piece.king:
            moves.update(self._traverse_left(row -1, max(row-3, -1), -1, piece.color, left))
            moves.update(self._traverse_right(row -1, max(row-3, -1), -1, piece.color, right))

        if piece.color == BLACK or piece.king:
            moves.update(self._traverse_left(row +1, min(row+3, ROWS), 1, piece.color, left))
            moves.update(self._traverse_right(row +1, min(row+3, ROWS), 1, piece.color, right))

        return moves

    def _traverse_left(self, start, stop, step, color, left, skipped=[]):
        moves = {}
        last = []
        for r in range(start, stop, step):
            if left < 0:
                break
            
            current = self.board[r][left]
            if current == 0:
                if skipped and not last:
                    break
                elif skipped:
                    moves[(r, left)] = last + skipped
                else:
                    moves[(r, left)] = last

                if last:
                    if step == -1:
                        row = max(r-3, 0)
                    else:
                        row = min(r+3, ROWS)

                    moves.update(self._traverse_left(r+step, row, step, color, left-1, skipped=last))
                    moves.update(self._traverse_right(r+step, row, step, color, left+1, skipped=last))
                break

            elif current.color == color:
                break
            else:
                last = [current]  

            left -= 1

        return moves


    def _traverse_right(self, start, stop, step, color, right, skipped=[]):
        moves = {}
        last = []
        for r in range(start, stop, step):
            if right >= COLS:
                break
            
            current = self.board[r][right]
            if current == 0:
                if skipped and not last:
                    break
                elif skipped:
                    moves[(r, right)] = last + skipped
                else:
                    moves[(r, right)] = last

                if last:
                    if step == -1:
                        row = max(r-3, 0)
                    else:
                        row = min(r+3, ROWS)

                    moves.update(self._traverse_left(r+step, row, step, color, right-1, skipped=last))
                    moves.update(self._traverse_right(r+step, row, step, color, right+1, skipped=last))
                break

            elif current.color == color:
                break
            else:
                last = [current]  

            right += 1

        return moves