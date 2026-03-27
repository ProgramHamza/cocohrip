from pathlib import Path
import pygame

WIDTH, HEIGHT = 800, 800
ROWS, COLS = 8, 8
SQUARE_SIZE = WIDTH//COLS

# rgb
RED = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREY = (220, 217, 209)
BROWN = (124, 84, 5)

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
CROWN = pygame.transform.scale(pygame.image.load(str(ASSETS_DIR / "crown.png")), (44, 25))
