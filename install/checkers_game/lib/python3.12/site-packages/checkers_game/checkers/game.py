from tkinter import NO
import pygame
from .board import Board
from ..constants import BLACK, RED, SQUARE_SIZE, WHITE, BLUE
from ..minimax.algorithm import WHITE, minimax
from copy import deepcopy

class Game:
    def __init__(self, win, boardDetection):
        self.win = win
        self.boardDetection = boardDetection
        self._init()

    def _init(self):
        self.aiMove = None
        self.aiPiece = None
        self.selected = None
        self.board = Board()
        self.turn = WHITE
        self.valid_moves = {}

    def update(self, boardDetected):

        if self.turn == BLACK:
            print("Hrac na tahu je: BLACK")
            # pass
        elif self.board.isBoardCreated == True:
            print("Hrac na tahu je: WHITE")
            # pass
        
        self.board.draw(self.win, boardDetected)

        self.change_turn()
        pygame.display.update()

                # print("hodnoty " + str(new_move[0] + str(new_move[1])))
        
        # if self.turn == BLACK and self.aiMove != None:
        #     self.show_ai_move()

        # self.change_turn()
        # pygame.display.update()

    def winner(self):
        return self.board.winner()

    def reset(self):
        self._init()

    def change_turn(self):
        if self.board.wasBoardChanged:
            self.valid_moves = {}
            if self.turn == BLACK:
                self.aiMove = None
                self.aiPiece = None
                self.turn = WHITE
            else:
                self.turn = BLACK
            
            self.board.wasBoardChanged = False

    def get_board(self):
        return self.board
    
    def show_ai_move(self):
        row, col = self.aiMove
        pygame.draw.circle(self.win, BLUE, (col * SQUARE_SIZE + SQUARE_SIZE//2, row * SQUARE_SIZE + SQUARE_SIZE//2), 15)
        
        self.aiPiece.draw(self.win)

        pygame.display.update()

    def wasItValidMove(self, boardDetected):
        oldBoard = deepcopy(self.get_board())
        if not oldBoard.board:
            return True
        

        self.board.draw(self.win, boardDetected)
        newBoard = deepcopy(self.get_board())
    
        moves = []
        newPossibleBoards = []

        for piece in oldBoard.get_all_pieces_by_color(WHITE):
            new_piece = deepcopy(piece)
            valid_moves = oldBoard.get_valid_moves(piece)
            for move, skip in valid_moves.items():
                temp_board = deepcopy(oldBoard)
                temp_piece = temp_board.get_piece(piece.row, piece.col)
                new_board, removed_pieces = self.simulate_move(temp_piece, move, temp_board, skip)
                moves.append([new_board, move, new_piece, removed_pieces])
                newPossibleBoards.append(new_board)

        valid_move_found  = self.find_valid_move(newBoard, newPossibleBoards)
        if not valid_move_found:
            # Move was not valid, revert to old board state
            # self.board = oldBoard.board
            self.board.black_left = oldBoard.black_left
            self.board.white_left = oldBoard.white_left
            self.board.black_kings = oldBoard.black_kings
            self.board.white_kings = oldBoard.white_kings
            self.board.white_kings = oldBoard.white_kings
            self.board.white_kings = oldBoard.white_kings
            self.board.wasBoardChanged = oldBoard.wasBoardChanged
            self.board.isBoardCreated = oldBoard.isBoardCreated
            self.board.board = oldBoard.board
        
            
            return False
        
        return True
    
    def find_valid_move(self, newBoard, possibleBoards):
        for possibleBoard in possibleBoards:
            if self.boards_are_equal(newBoard.board, possibleBoard.board):
                # Found a match, return corresponding move or True to indicate validity
                return True  # Or return the specific move if you need that information
        return False
    
    def boards_are_equal(self, board1, board2):
        for row in range(len(board1)):
            for col in range(len(board1[row])):
                piece1 = board1[row][col]
                piece2 = board2[row][col]
                if bool(piece1) != bool(piece2):  # Checks if both are either pieces or empty
                    return False
                if piece1 and piece2:  # If both positions are not empty
                    if not piece1.equals(piece2):
                        return False
        return True
    
    
    def simulate_move(self, piece, move, board, skip):
        board.move(piece, move[0], move[1])
        removed_pieces = []
        if skip:
            for skip_piece in skip:
                board.remove(skip_piece)
                removed_pieces.append(skip_piece)
        
        return board, removed_pieces