import sys
from pathlib import Path
import pygame
from copy import deepcopy

# Ensure package root (folder that contains __init__.py) is on sys.path
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from .constants import WIDTH, HEIGHT, SQUARE_SIZE, WHITE, BLACK, BLUE, RED
from .checkers.board import Board
from .minimax.algorithm import minimax


FPS = 60
HIGHLIGHT_COLOR = (0, 200, 0)
TEXT_COLOR = (245, 245, 245)
BACKGROUND_COLOR = (30, 30, 30)
STATUS_PANEL_HEIGHT = 60
WINDOW_HEIGHT = HEIGHT + STATUS_PANEL_HEIGHT


def initial_board_matrix():
    """Return the standard 8x8 starting layout encoded like camera detection."""
    layout = []
    for row in range(8):
        row_values = []
        for col in range(8):
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


def get_row_col_from_mouse(pos):
    x, y = pos
    row = (y - STATUS_PANEL_HEIGHT) // SQUARE_SIZE
    col = x // SQUARE_SIZE
    return int(row), int(col)


def draw_valid_moves(win, moves):
    for row, col in moves:
        center = (
            col * SQUARE_SIZE + SQUARE_SIZE // 2,
            STATUS_PANEL_HEIGHT + row * SQUARE_SIZE + SQUARE_SIZE // 2,
        )
        pygame.draw.circle(win, HIGHLIGHT_COLOR, center, 12)


def draw_selected(win, row, col):
    center = (
        col * SQUARE_SIZE + SQUARE_SIZE // 2,
        STATUS_PANEL_HEIGHT + row * SQUARE_SIZE + SQUARE_SIZE // 2,
    )
    pygame.draw.circle(win, BLUE, center, 18, width=3)


def draw_status_panel(win, font, turn, depth, message):
    panel_rect = pygame.Rect(0, 0, WIDTH, STATUS_PANEL_HEIGHT)
    pygame.draw.rect(win, BACKGROUND_COLOR, panel_rect)

    turn_text = "Your turn" if turn == WHITE else "AI thinking"
    text_surface = font.render(f"{turn_text} | AI depth {depth} | {message}", True, TEXT_COLOR)
    win.blit(text_surface, (12, 18))


def draw_window(win, board, valid_moves, selected, turn, depth, message):
    win.fill(BACKGROUND_COLOR)
    font = pygame.font.SysFont("arial", 20)
    draw_status_panel(win, font, turn, depth, message)

    # shift board drawing below status panel
    board_surface = win.subsurface(pygame.Rect(0, STATUS_PANEL_HEIGHT, WIDTH, HEIGHT))
    board.draw_squares(board_surface)

    for row in range(8):
        for col in range(8):
            piece = board.board[row][col]
            if piece != 0:
                piece.draw(board_surface)

    if selected is not None:
        draw_selected(win, selected.row, selected.col)

    draw_valid_moves(win, valid_moves.keys())
    pygame.display.update()


def reset_board():
    board = Board()
    board.create_board(initial_board_matrix())
    return board


def perform_player_move(board, piece, destination, valid_moves):
    skip = valid_moves.get(destination)
    if skip is None:
        return False
    row, col = destination
    board.move(piece, row, col)
    for captured in skip:
        board.remove(captured)
    return True


def perform_ai_move(board, depth):
    value, new_board, new_move, new_piece, removed = minimax(deepcopy(board), depth, True, None)
    if new_board is None or new_move is None or new_piece is None:
        return board, False
    return new_board, True


def run_game():
    pygame.init()
    pygame.display.set_caption("Checkers - Local Play")
    win = pygame.display.set_mode((WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    board = reset_board()
    turn = WHITE
    selected_piece = None
    valid_moves = {}
    ai_depth = 3
    status_message = "Click a white piece to move"
    game_over = None

    running = True
    while running:
        clock.tick(FPS)
        draw_window(win, board, valid_moves, selected_piece, turn, ai_depth, status_message)

        if game_over is None:
            winner = board.winner()
            if winner is not None:
                game_over = winner
                status_message = f"{winner} wins! Press R to restart"

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    board = reset_board()
                    turn = WHITE
                    selected_piece = None
                    valid_moves = {}
                    status_message = "Game reset"
                    game_over = None
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                    ai_depth = int(event.unicode)
                    status_message = f"AI depth set to {ai_depth}"
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and game_over is None:
                pos = pygame.mouse.get_pos()
                if pos[1] < STATUS_PANEL_HEIGHT:
                    continue
                row, col = get_row_col_from_mouse(pos)
                if row < 0 or row >= 8 or col < 0 or col >= 8:
                    continue

                if turn == WHITE:
                    piece = board.get_piece(row, col) if board.board else None
                    if selected_piece and (row, col) in valid_moves:
                        moved = perform_player_move(board, selected_piece, (row, col), valid_moves)
                        if moved:
                            turn = BLACK
                            selected_piece = None
                            valid_moves = {}
                            status_message = "AI is thinking..."
                            draw_window(win, board, valid_moves, selected_piece, turn, ai_depth, status_message)
                            pygame.display.flip()
                            board, ai_moved = perform_ai_move(board, ai_depth)
                            turn = WHITE
                            status_message = "Your turn" if ai_moved else "AI had no moves"
                        else:
                            status_message = "Illegal destination"
                    elif piece != 0 and piece.color == WHITE:
                        selected_piece = piece
                        valid_moves = board.get_valid_moves(piece)
                        if not valid_moves:
                            status_message = "No moves for that piece"
                        else:
                            status_message = "Select a destination"
                    else:
                        selected_piece = None
                        valid_moves = {}
                        status_message = "Select one of your pieces"

        if game_over is not None:
            continue

    pygame.quit()


def main():
    run_game()


if __name__ == "__main__":
    main()
