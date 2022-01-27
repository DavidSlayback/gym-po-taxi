import numpy as np

N_PIXEL_CELL = 16
N_PIXEL_WALL = int(N_PIXEL_CELL / 4)

BLACK = np.zeros(3, dtype=np.uint8)
WHITE = np.full_like(BLACK, 255)
GRAY = np.full_like(BLACK, 128)
DARK_GRAY = np.full_like(BLACK, 64)
LIGHT_GRAY = np.full_like(BLACK, 192)
RED = np.array([128, 0, 0], dtype=np.uint8)
GREEN = np.array([0, 128, 0], dtype=np.uint8)
BLUE = np.array([0, 0, 128], dtype=np.uint8)
YELLOW = np.array([128, 128, 0], dtype=np.uint8)
PURPLE = np.array([128, 0, 128], dtype=np.uint8)
TEAL = np.array([0, 128, 128], dtype=np.uint8)

def lighten(color_or_cell: np.ndarray, amount: int = 64, in_place: bool = True, saturate: bool = False):
    """Highlight a color. Could also darken it"""
    mask = color_or_cell > 0 if not saturate else color_or_cell >= 0
    if in_place:
        color_or_cell[mask] = np.clip(color_or_cell[mask].astype('int') + amount, 0, 255)
        return color_or_cell
    else:
        new_color = color_or_cell.copy()
        new_color[mask] = np.clip(color_or_cell[mask].astype('int') + amount, 0, 255)
        return new_color


def template(cell: np.ndarray, color: np.ndarray) -> np.ndarray:
    nc = cell.copy()
    nc[:, :] = color
    return nc


BASE_CELL = np.zeros((16, 16, 3), dtype=np.uint8)  # Grid cells are 16x16
BASE_GRID_CELL = template(BASE_CELL, GRAY)  # Floor is medium gray

# Taxi walls need special handling
TAXI_VERTICAL_WALL = np.zeros((4, 16, 3), dtype=np.uint8)
TAXI_VERTICAL_DIVIDER = lighten(TAXI_VERTICAL_WALL, 64, False, True)  # Dark gray
TAXI_HORIZONTAL_WALL = np.zeros((16, 4, 3), dtype=np.uint8)
TAXI_CORNER = np.zeros((4, 4, 3), dtype=np.uint8)

# Agent (fr, Carflag) and full taxi are green
AGENT = template(BASE_CELL, GREEN)
TAXI_FULL = AGENT

# Goals (taxi destination and fourrooms goal and heaven) are blue
GOAL = template(BASE_CELL, BLUE)
# Taxi passenger location is purple
PASSENGER = template(BASE_CELL, PURPLE)
# Stairs and hell are red
STAIR = template(BASE_CELL, RED)
HELL = STAIR
# Taxi destinations are white
DESTINATION = template(BASE_CELL, WHITE)
