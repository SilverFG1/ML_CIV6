"""Simulation constants. Importing this module never initializes graphics."""

import itertools
import sys
from pathlib import Path

MAP_WIDTH = MAP_HEIGHT = 8
HEX_SIZE = CELL_WIDTH = CELL_HEIGHT = 96
EDGE_OFFSET = 24
GAME_FPS = 10
LOC_CITY = (4, 4)
COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_RED = (210, 60, 60)
COLOR_PURPLE = (184, 157, 207)
COLOR_LIGHT_GREY = (225, 225, 225)
COLOR_DEFAULT_BG = (28, 39, 48)

DIRECTIONS = ('NE', 'E', 'SE', 'SW', 'W', 'NW', 'SPACE')
directions = list(DIRECTIONS)
MOVEMENT_DIR = {
    'NE': {'EVEN': (0, -1), 'ODD': (1, -1)},
    'E': {'EVEN': (1, 0), 'ODD': (1, 0)},
    'SE': {'EVEN': (0, 1), 'ODD': (1, 1)},
    'SW': {'EVEN': (-1, 1), 'ODD': (0, 1)},
    'W': {'EVEN': (-1, 0), 'ODD': (-1, 0)},
    'NW': {'EVEN': (-1, -1), 'ODD': (0, -1)},
    'SPACE': {'EVEN': (0, 0), 'ODD': (0, 0)},
}
MOVEMENT_ONE_UNIT = list(DIRECTIONS)
MOVEMENT_TWO_UNITS = list(itertools.product(DIRECTIONS, repeat=2))
MOVEMENT_THREE_UNITS = list(itertools.product(DIRECTIONS, repeat=3))
HEX_LOCATIONS = [(x, y) for y in range(MAP_HEIGHT) for x in range(MAP_WIDTH)
                 if (x, y) != LOC_CITY]


def resource_path(relative_path):
    """Locate assets both from source and a PyInstaller bundle."""
    base = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent))
    return str(base / relative_path)


def movement_delta(direction, y):
    return MOVEMENT_DIR[direction]['ODD' if y % 2 else 'EVEN']
