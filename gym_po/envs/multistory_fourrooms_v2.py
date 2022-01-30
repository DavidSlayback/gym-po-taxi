import time

from enum import IntEnum, auto
from typing import List, Optional, Union, Tuple

import gym
from gym.utils.seeding import np_random
import numpy as np
import cv2

# Grid values for different squares
from gym.spaces import Discrete
from gym.vector.utils import batch_space


class GR_CNST(IntEnum):
    wall = auto()
    empty = auto()
    stair_up = auto()
    stair_down = auto()
    agent = auto()
    goal = auto()

FR_COLOR_MAP = {
    GR_CNST.wall: np.array([0,0,0], dtype=np.uint8),
    GR_CNST.empty: np.array([128,128,128], dtype=np.uint8),
    GR_CNST.stair_up: np.array([128,0,128], dtype=np.uint8),
    GR_CNST.stair_down: np.array([0,128,128], dtype=np.uint8),
    GR_CNST.agent: np.array([0,128,0], dtype=np.uint8),
    GR_CNST.goal: np.array([0,0,128], dtype=np.uint8),
}

# Basic four rooms layout, in x,y as would be in images
FR_LAYOUT = """\
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""

# As numpy
FR_LAYOUT_NP = np.array([list(map(lambda c: GR_CNST.wall if c=='w' else GR_CNST.empty, line)) for line in FR_LAYOUT.splitlines()], dtype=np.uint8)
def gen_layout(base_grid: np.ndarray = FR_LAYOUT_NP, grid_z: int = 1):
    """Generate multistory layout based on some base floor grid

    Assumes walls all around, places stairs in NE and SW
    """
    y, x = base_grid.shape
    NE = (1, x - 2)
    SW = (y - 2, 1)  # Generate floors with stairs
    if not (grid_z - 1): return base_grid.copy()[None, ...], NE, SW  # Just add z dim
    bfloor, mfloor, tfloor = (base_grid.copy() for _ in range(3))
    bfloor[NE] = GR_CNST.stair_up; tfloor[SW] = GR_CNST.stair_down; mfloor[SW] = GR_CNST.stair_down; mfloor[NE] = GR_CNST.stair_up
    return np.stack((bfloor, *(mfloor.copy() for _ in range(grid_z - 2)), tfloor), axis=0), NE, SW  # Stack on z-dim

def grid_to_render_coordinates(yx_coords: np.ndarray, cell_pixel_size: int = 16):
    """Convert grid coordinates (e.g. 0-12 in 13x13) to render coordinates (0-208)"""
    if yx_coords.ndim == 1:
        return np.mgrid[:cell_pixel_size, :cell_pixel_size] + (yx_coords * cell_pixel_size)[...,None,None]
    else:  # Multiple coordinates
        return np.mgrid[:cell_pixel_size, :cell_pixel_size][None,...] + (yx_coords.T * cell_pixel_size)[..., None, None]

def grid_to_many_render_coordinates(myx_coords: np.ndarray, cell_pixel_size: int = 16):
    """Convert many grid coordinates (e.g. 0-12 in 13x13) to render coordinates (0-208)"""
    return [tuple(c) for c in (np.mgrid[:cell_pixel_size, :cell_pixel_size][None, ...] + (myx_coords.T * cell_pixel_size)[..., None, None])]

def gen_base_render_layout(layout: np.ndarray = FR_LAYOUT_NP[None,...], cell_pixel_size: int = 16) -> List[np.ndarray]:
    """Create numpy rendering grids based on layout

    Assume incoming layout is y,x w.r.t. image, we'll want to render as x,y
    """
    z, y, x = layout.shape
    base_render = np.full((y*cell_pixel_size, x*cell_pixel_size, 3), FR_COLOR_MAP[GR_CNST.empty], dtype=np.uint8)  # Base floor to render
    wall_coord = np.array(np.where(layout[0] == GR_CNST.wall))
    r_wall_coord = grid_to_many_render_coordinates(wall_coord)
    for c in r_wall_coord:
        base_render[c] = FR_COLOR_MAP[GR_CNST.wall]
    if not (z-1): return [base_render]
    bfloor, mfloor, tfloor = (base_render.copy() for _ in range(3))
    bfloor[tuple(grid_to_render_coordinates(np.array(np.where(layout[0] == GR_CNST.stair_up)).squeeze()))] = FR_COLOR_MAP[GR_CNST.stair_up]
    tfloor[tuple(grid_to_render_coordinates(np.array(np.where(layout[-1] == GR_CNST.stair_down)).squeeze()))] = FR_COLOR_MAP[GR_CNST.stair_down]
    if z == 2: return [bfloor, tfloor]
    else:
        mfloor[tuple(grid_to_render_coordinates(np.array(np.where(layout[0] == GR_CNST.stair_up)).squeeze()))] = FR_COLOR_MAP[GR_CNST.stair_up]
        mfloor[tuple(grid_to_render_coordinates(np.array(np.where(layout[-1] == GR_CNST.stair_down)).squeeze()))] = FR_COLOR_MAP[GR_CNST.stair_down]
        return [bfloor, mfloor, tfloor]


def add_agent_and_goal_to_render(render_map: np.ndarray, ayx_coord: np.ndarray, gyx_coord: Optional[np.ndarray] = None):
    """Single map render only"""
    m = render_map.copy()  # Not in-place
    if gyx_coord is not None: m[tuple(grid_to_render_coordinates(ayx_coord))] = FR_COLOR_MAP[GR_CNST.agent]
    m[tuple(grid_to_render_coordinates(gyx_coord))] = FR_COLOR_MAP[GR_CNST.goal]
    return m

# Action stuff
def action_probability_matrix(action_failure_probability: float = (1./3), action_n: int = 4):
    probs = np.full((action_n, action_n), action_failure_probability / (action_n - 1), dtype=np.float64)
    np.fill_diagonal(probs, 1 - action_failure_probability)
    return probs

def vectorized_multinomial(selected_prob_matrix: np.ndarray, random_numbers: np.ndarray):
    """Vectorized sample from [B,N] probabilitity matrix
    Lightly edited from https://stackoverflow.com/a/34190035/2504700
    Args:
        selected_prob_matrix: (Batch, p) size probability matrix (i.e. T[s,a] or O[s,a,s']
        random_numbers: (Batch,) size random numbers from np.random.rand()
    Returns:
        (Batch,) size sampled integers
    """
    s = selected_prob_matrix.cumsum(axis=1)  # Sum over p dim for accumulated probability
    return (s < np.expand_dims(random_numbers, axis=-1)).sum(axis=1)  # Returns first index where random number < accumulated probability

class MultistoryFourRoomsEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    """Discrete multistory fourrooms environment"""
    def __init__(self, num_envs: int, grid_z: int = 1, time_limit: int = 10000, seed=None,
                 fixed_goal: Union[Tuple[int, int, int], int] = 0, action_failure_probability: float = (1./3),
                 wall_reward: float = 0., goal_z=None, agent_z=None):
        # Preliminaries
        self.num_envs = num_envs
        self.is_vector_env = True
        self.time_limit = time_limit
        self.wall_reward = wall_reward
        agent_z = agent_z or 0
        goal_z = goal_z or (grid_z-1)
        self.gz, self.az = goal_z, agent_z

        # Actions
        self.single_action_space = Discrete(4)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.aprob_with_failure = action_probability_matrix(action_failure_probability, 4)

        # Grid
        self.grid, self.ne, self.sw = gen_layout(grid_z=grid_z)  # FR Layout
        y,x = self.grid.shape[1:]
        self.grid_z = grid_z
        empty_locs = np.array((self.grid == GR_CNST.empty).nonzero())  # Where can things spawn
        valid_locs = np.array((self.grid != GR_CNST.wall).nonzero())  # Where can agent be
        self.encode = lambda zyx_coord: np.ravel_multi_index(zyx_coord, (grid_z, y, x))
        self.decode = lambda i: np.unravel_index(i, (grid_z, y, x))
        self.flat_empty_locs = self.encode(empty_locs)
        self.flat_valid_locs = self.encode(valid_locs)
        self.goal_locs = empty_locs[:, np.isin(empty_locs[0], goal_z, True)].T
        self.agent_locs = empty_locs[:, np.isin(empty_locs[0], agent_z, True)].T
        self.actions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])  # Up, down, left, right
        self.render_maps = gen_base_render_layout(self.grid)  # Render maps for speed

        # Observations
        self._ns = valid_locs.shape[-1]
        self.single_observation_space = Discrete(self._ns)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.fcoord_to_ecoord = np.isin(np.arange(self.grid.size), self.flat_valid_locs, True).cumsum() - 1  # To discrete
        self.time_limit = time_limit
        self._viewer = None

        self.seed(seed)
        if fixed_goal:
            self.goal_fn = lambda b: fixed_goal
        else: self.goal_fn = lambda b: self.goal_locs[self.rng.randint(len(self.goal_locs), size=b, dtype=np.int64)]
        self.agent = np.ones((num_envs, 3), int)  # zyx coordinates of each agent
        self.goal = np.ones((num_envs, 3), int)  # zyx coordinates of each goal
        self.elapsed = np.zeros((num_envs,), int)  # Steps in each env

    def seed(self, seed: Optional[int] = None):
        self.rng, seed = np_random(seed)
        return seed

    def render(self, mode="human"):
        azyx_coord = self.agent[0]  # First agent
        z = azyx_coord[0]
        gzyx_coord = self.goal[0]
        if self.grid_z == 2 and z == 1: m = self.render_maps[1]
        elif self.grid_z > 2 and z != 0 and azyx_coord[0] != self.grid_z - 1: m = self.render_maps[2]
        else: m = self.render_maps[0]
        img = add_agent_and_goal_to_render(m, azyx_coord[1:], gzyx_coord[1:] if z == gzyx_coord[0] else None)
        if mode == 'rgb' or mode == 'rgb_array': return img
        else:
            import pygame
            if self._viewer is None:
                pygame.init()
                self._viewer = pygame.display.set_mode(img.shape[:-1])
            sfc = pygame.surfarray.make_surface(img)
            self._viewer.blit(sfc, (0,0))
            pygame.display.update()
            return img

    def reset(self):
        self._reset_mask(np.ones(self.num_envs, bool))
        return self._obs()

    def _reset_mask(self, mask: np.ndarray):
        b = mask.sum()
        if b:
            self.goal[mask] = self.goal_fn(b)
            # Make sure agent doesn't start on goal
            self.agent[mask] = self.agent_locs[self.rng.randint(len(self.agent_locs), size=b, dtype=np.int64)]
            while (m := mask & (self.agent == self.goal).all(-1)).any():
                self.agent[m] = self.agent_locs[self.rng.randint(len(self.agent_locs), size=m.sum(), dtype=np.int64)]
            self.elapsed[mask] = 0

    def _obs(self):
        return self.fcoord_to_ecoord[self.encode(self.agent.T)]


test = MultistoryFourRoomsEnv(16)
o = test.reset()
a = test.render()
print(o)