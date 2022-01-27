import numpy as np
from render_utils import (TAXI_VERTICAL_WALL, TAXI_HORIZONTAL_WALL, TAXI_VERTICAL_DIVIDER, TAXI_CORNER,
                          BASE_GRID_CELL, N_PIXEL_CELL, N_PIXEL_WALL, GRAY,
                          DESTINATION)
from PIL import Image

# Base map for Taxi
MAP = np.asarray([
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
], dtype='c').astype('str')

# Where are we thin?
H_IDX = [0, 6]
V_IDX = [0, 2, 4, 6, 8, 10]
RENDER_MAP = np.zeros((5 * N_PIXEL_CELL + len(V_IDX) * N_PIXEL_WALL, 5 * N_PIXEL_CELL + len(H_IDX) * N_PIXEL_WALL, 3), dtype=np.uint8)  # x,y
RENDER_MAP[:, :] = GRAY
start_y = 0
for y in range(MAP.shape[0]):
    start_x = 0
    hwall = (y in H_IDX)
    for x in range(MAP.shape[1]):
        vwall = (x in V_IDX)
        corner = (x in (0,11)) and hwall
        spacer = hwall and vwall
        if corner:
            RENDER_MAP[start_x:start_x + N_PIXEL_WALL, start_y:start_y + N_PIXEL_WALL] = TAXI_CORNER
            start_x += N_PIXEL_WALL
        elif spacer:
            RENDER_MAP[start_x:start_x + N_PIXEL_WALL, start_y:start_y + N_PIXEL_WALL] = TAXI_CORNER[0,0]
            start_x += N_PIXEL_WALL
        elif hwall:  # Bottom or top, not a corner
            RENDER_MAP[start_x:start_x + N_PIXEL_CELL, start_y:start_y + N_PIXEL_WALL] = TAXI_HORIZONTAL_WALL
            start_x += N_PIXEL_CELL
        elif vwall:
            block = TAXI_VERTICAL_DIVIDER if MAP[y,x] == ':' else TAXI_VERTICAL_WALL
            RENDER_MAP[start_x:start_x + N_PIXEL_WALL, start_y:start_y + N_PIXEL_CELL] = block
            start_x += N_PIXEL_WALL
        else:
            if MAP[y, x] in ('R', 'Y', 'G', 'B'):
                RENDER_MAP[start_x:start_x + N_PIXEL_CELL, start_y: start_y + N_PIXEL_CELL] = DESTINATION
            start_x += N_PIXEL_CELL
    start_y = start_y + N_PIXEL_WALL if hwall else start_y + N_PIXEL_CELL
test = Image.fromarray(RENDER_MAP.swapaxes(0,1))
test.save('test.png')
