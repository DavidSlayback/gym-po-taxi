__all__ = ['MAP_BASE', 'BOTTOM_MAP', 'MID_MAP', 'TOP_MAP', 'get_populated_render_map']

from typing import Optional

import numpy as np
from render_utils import N_PIXEL_CELL, GRAY, RED, GREEN
WALL = 0
EMPTY = 1
STAIR = 2
GOAL = 3

# Basic 13x13 FourRooms grid. 0 is wall. 1 is empty
MAP_BASE = np.array([
    [WALL]*13,
    [WALL]+([EMPTY]*5) + [WALL] + ([EMPTY]*5) + [WALL],
    [WALL]+([EMPTY]*5) + [WALL] + ([EMPTY]*5) + [WALL],
    [WALL]+([EMPTY]*11) + [WALL],
    [WALL]+([EMPTY]*5) + [WALL] + ([EMPTY]*5) + [WALL],
    [WALL]+([EMPTY]*5) + [WALL] + ([EMPTY]*5) + [WALL],
    [WALL, WALL, EMPTY] + ([WALL]*4) + ([EMPTY]*5) + [WALL],
    [WALL] + ([EMPTY]*5) + ([WALL]*3) + [EMPTY] + ([WALL]*3),
    [WALL]+([EMPTY]*5) + [WALL] + ([EMPTY]*5) + [WALL],
    [WALL]+([EMPTY]*5) + [WALL] + ([EMPTY]*5) + [WALL],
    [WALL]+([EMPTY]*11) + [WALL],
    [WALL]+([EMPTY]*5) + [WALL] + ([EMPTY]*5) + [WALL],
    [WALL]*13,
], dtype=int)  # Row-major, so up would be [-1, 0]
NE = (1,11)  # Upstairs
SW = (11,1)  # Downstairs
MID_MAP = MAP_BASE.copy()
MID_MAP[NE], MID_MAP[SW] = STAIR, STAIR
BOTTOM_MAP = MAP_BASE.copy()
BOTTOM_MAP[NE] = STAIR
TOP_MAP = MAP_BASE.copy()
TOP_MAP[NE] = STAIR

# def map_coord_to_image_coord(yx_coord: np.ndarray) -> np.ndarray:
#     return yx_coord.T * N_PIXEL_CELL

# images are (0,0) (w,h) from top, so flip
RENDER_MAP_BASE = np.zeros((MAP_BASE.shape[1] * N_PIXEL_CELL, MAP_BASE.shape[0] * N_PIXEL_CELL, 3), np.uint8)
w_idxs = [(i*N_PIXEL_CELL, i*N_PIXEL_CELL+N_PIXEL_CELL) for i in range(MAP_BASE.shape[1])]
h_idxs = [(i*N_PIXEL_CELL, i*N_PIXEL_CELL+N_PIXEL_CELL) for i in range(MAP_BASE.shape[0])]
for y, hidx in enumerate(h_idxs):
    for x, widx in enumerate(w_idxs):
        if MAP_BASE[y, x] != WALL:
            RENDER_MAP_BASE[hidx[0]:hidx[1], widx[0]:widx[1]] = GRAY

BOTTOM_RENDER_MAP_BASE = RENDER_MAP_BASE.copy()
BOTTOM_RENDER_MAP_BASE[h_idxs[1][0]:h_idxs[1][1], w_idxs[11][0]:w_idxs[11][1]] = RED
TOP_RENDER_MAP_BASE = RENDER_MAP_BASE.copy()
TOP_RENDER_MAP_BASE[h_idxs[11][0]:h_idxs[11][1], w_idxs[1][0]:w_idxs[1][1]] = RED
MID_RENDER_MAP_BASE = BOTTOM_RENDER_MAP_BASE.copy()
MID_RENDER_MAP_BASE[h_idxs[11][0]:h_idxs[11][1], w_idxs[1][0]:w_idxs[1][1]] = RED

def get_render_map(grid_z: int, n_floors: int = 1) -> np.ndarray:
    """Return appropriate base render map given specified floor"""
    if n_floors == 1: return RENDER_MAP_BASE.copy()  # No stairs
    if not grid_z: return BOTTOM_RENDER_MAP_BASE.copy()
    elif grid_z == n_floors - 1: return TOP_RENDER_MAP_BASE.copy()
    else: return MID_RENDER_MAP_BASE.copy()


def get_populated_render_map(agent_yxz: np.ndarray, goal_yxz: Optional[np.ndarray] = None, obs_n: int = 3, n_floors: int = 1):
    """Create render image with agent and goal (if visible)"""
    y, x, z = agent_yxz
    grid = get_render_map(z)  # Base grid
    grid[h_idxs[y][0]:h_idxs[y][1], w_idxs[x][0]:w_idxs[x][1]] = GREEN



