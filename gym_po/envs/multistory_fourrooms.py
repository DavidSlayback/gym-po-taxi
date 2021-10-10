

import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
from gym.vector.utils import batch_space
from gym.utils.seeding import np_random
from typing import Optional

WALL = 0
EMPTY = 1
STAIR = 2
GOAL = 3

# Up, right, down, left
ACTIONS = np.array([
    [-1, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [0, -1, 0]
], dtype=int)

# Basic 13x13 FourRooms grid. 0 is wall. 1 is empty
MAP = np.array([
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
], dtype=int)
NE = (1, 11)  # NE stairs are upstairs
SW = (11, 1)  # SW stairs are downstairs

# Add stairs to bottom, mid, and top floors
B_MAP, M_MAP, T_MAP = [MAP.copy()] * 3
B_MAP[NE] = STAIR; M_MAP[NE] = M_MAP[SW] = STAIR; T_MAP[SW] = STAIR

if __name__ == "__main__":
    print(3)
