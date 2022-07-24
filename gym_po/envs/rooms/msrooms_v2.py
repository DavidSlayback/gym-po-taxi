from typing import Tuple, Sequence, Union, Callable

import numpy as np
from dotsi import DotsiDict
from .render_utils import COLORS
from gym.spaces import Discrete, Box, Space
from .observations import get_number_discrete_states_and_conversion
from .actions import *

# 13x13 FourRooms, upstairs and downstairs locations marked by "U" and "D"
BASE_FOURROOMS_MAP_WITH_STAIRS = '''xxxxxxxxxxxxx
                                    x11111x00000x
                                    x11111x00000x
                                    x11111100000x
                                    x11111x00000x
                                    x11111x00000x
                                    xx2xxxx00000x
                                    x22222xxx3xxx
                                    x22222x33333x
                                    x22222x33333x
                                    x22222333333x
                                    x22222x33333x
                                    xxxxxxxxxxxxx'''
WALL_CHAR = 'x'
# Fixed locations
END = (7, 9)  # East hallway
START = NW = (1, 1)  # NW Corner
SW = (11, 1)
NE = (1, 11)

# Constant integers for each object
GR_CNST = DotsiDict({
    'wall': 0,
    'goal': 1,
    'stair_up': 2,
    'stair_down': 3
})


# Constant colors, can be indexed by values above
GR_CNST_COLORS = DotsiDict({
    'wall': COLORS.black,
    'empty': COLORS.gray_dark,
    'agent': COLORS.green,
    'goal': COLORS.blue,
    'stair_up': COLORS.gray_light,
    'stair_down': COLORS.gray
})


def get_grid_obs(agent_zyx: np.ndarray, grid: np.ndarray, goal_zyx: np.ndarray, n: int = 3) -> np.ndarray:
    """Return grid observation

    Args:
        agent_yx: (z, y, x) coordinate of agent(s) [B, 3]
        grid: (z, y, x) numpy grid
        goal_yx: (z, y, x) goal location(s) [B, 3]
        n: 8 or 4
    Returns:
        obs: See constants
    """
    offset = n // 2  # Center on agent
    mg = np.mgrid[0:1, n, :n] - offset  # 3x1xNxN
    coords = (agent_zyx[...,None, None] + mg[None, ...]).swapaxes(0,1)
    # All invalid coords should point to a wall (i.e., (0,0))
    coords[:, (coords[0] < 0) | (coords[1] < 0) | (coords[0] >= grid.shape[0]) | (coords[1] >= grid.shape[1])] = 0
    is_goal = (goal_zyx.swapaxes(0,1)[..., None, None] == coords).all(0)
    squares = grid[tuple(coords)] + 1
    squares[squares > 0] = 1
    squares[is_goal] = 2
    return squares

def layout_to_np(layout: str) -> np.ndarray:
    """Convert layout string to numpy char array"""
    return np.asarray([t.strip() for t in layout.splitlines()], dtype='c').astype('U')


def np_to_grid(np_layout: np.ndarray) -> Tuple[np.ndarray, int]:
    """Convert numpy char array to state-abstracted integer grid, also return number of states"""
    state_aliases = np.unique(np_layout)
    state_aliases_without_wall = np.delete(state_aliases, np.nonzero(state_aliases == WALL_CHAR))
    first_val = max(GR_CNST.values())
    state_alias_values = np.arange(first_val, len(state_aliases_without_wall)+first_val)
    grid = np.full_like(np_layout, 0, dtype=int)  # Walls are 0
    for i, a in zip(state_alias_values, state_aliases_without_wall):
        grid[np_layout == a] = i  # State aliases 1,2,3,4 in FourRooms
    return grid, len(state_alias_values)


def generate_layout_and_img(map: str = BASE_FOURROOMS_MAP_WITH_STAIRS, grid_z: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Return full (z,y,x) map layout and unscaled (z,y,x,3) rgb image

    Args:
        map: Raw string map
        grid_z: Number of floors
    Returns:
        cnst_layout: (z,y,x) grid layout with room abstractions and stairs
        img_map (z,y,x,3) unscaled img
    """
    state_map, n_states = np_to_grid(layout_to_np(map))  # Bordered map with state abstractions, no stairs
    cnst_layout = np.stack(tuple(state_map + (i * n_states) for i in range(grid_z)), axis=0)
    cnst_layout[:, state_map == GR_CNST.wall] = GR_CNST.wall
    if grid_z >= 2:  # Middle floors have both sets of stairs, top and bottom have down and up
        cnst_layout[0:-1, NE[0], NE[1]] = GR_CNST.stair_up
        cnst_layout[1:, SW[0], SW[1]] = GR_CNST.stair_down
    img_map = np.zeros_like(cnst_layout, shape=(*cnst_layout.shape, 3), dtype=np.uint8)  # Unscaled image version
    for (k, v) in GR_CNST.items():
        img_map[cnst_layout == v] = GR_CNST_COLORS[k]
    return cnst_layout, img_map


def get_hansen_vector_obs(agent_yx: np.ndarray, grid: np.ndarray, goal_yx: Optional[np.ndarray] = None, hansen_n: int = 8) -> np.ndarray:
    """Same as above, but a vector representation (like the grid obs, but flattened)

    Args:
        agent_yx: (z, y,x) coordinate of agent(s) [B,3]
        grid: (z, y,x) numpy grid
        goal_yx (z, y,x) goal location(s) [B, 3]
        n: 8 or 4
    Returns:
        Obs (Constants)
    """
    a = ACTIONS_CARDINAL if hansen_n == 4 else ACTIONS_ORDINAL
    a = a[None, :]
    coords = agent_yx[:, None] + a
    squares = grid[tuple(coords.transpose(2,0,1))]
    squares += 1
    squares[squares > 0] = 1  # Empty squares
    if goal_yx is not None:
        is_goal = (goal_yx[:, None] == coords).all(-1)
        squares[is_goal] = 2  # Add goal
    return squares


def get_observation_space_and_function(obs_type: str, ms_grid: np.ndarray, obs_n: int = 3) -> Tuple[Space, Callable[[np.ndarray, np.ndarray], np.ndarray]]:
    """Return Gym Space and observation function (takes agent and goal positions (zyx) as input)"""
    is_vector = 'vector' in obs_type  # Return a vector or a scalar? In practice, scalar can get huge
    has_goal = 'goal' in obs_type  # Should goal information be included?
    a_max = np.array(ms_grid.shape) - 2; a_max[0] += 1  # Max agent coordinate in 3-D
    a_min = np.array([0,1,1])  # Min agent coordinate in 3-D
    if 'room' in obs_type:
        assert not is_vector
        offset = len(GR_CNST)
        n = ms_grid.max() - offset  # Number of distinct rooms, offset by number of other state types
        if has_goal:
            space = Discrete(int(n ** 2))
            obs = lambda azyx, gzyx: (ms_grid[tuple(azyx.T)] - offset) + n * (ms_grid[tuple(gzyx.T)] - offset)
        else:
            space = Discrete(int(n))
            obs = lambda azyx, gzyx: ms_grid[tuple(azyx.T)]
    elif 'mdp' in obs_type:  # Fully observable (if goal is fixed or provided)
        if is_vector:  # Vector obs of position
            if has_goal:
                space = Box(a_min, np.tile(a_max, 2), (6,), dtype=int)
                obs = lambda azyx, gzyx: np.concatenate((azyx, gzyx), -1)
            else:
                space = Box(a_min, a_max, (3,), dtype=int)
                obs = lambda azyx, gzyx: azyx
        else:  # Discrete state for agent (and maybe goal)
            n, state_grid = get_number_discrete_states_and_conversion(ms_grid - 1)
            if has_goal:
                space = Discrete(int(n ** 2))
                obs = lambda azyx, gzyx: state_grid[tuple(azyx.T)] + n * state_grid[tuple(gzyx.T)]
            else:
                space = Discrete(int(n))
                obs = lambda azyx, gzyx: state_grid[tuple(azyx.T)]
    elif 'hansen' in obs_type:
        base_n = 8 if '8' in obs_type else 4
        if is_vector:
            if has_goal:
                space = Box(0, 2, (base_n,), dtype=int)
                obs = lambda azyx, gzyx: get_hansen_vector_obs(ayx, grid, gyx, base_n)
            else:
                space = Box(0, 1, (base_n,), dtype=int)
                obs = lambda azyx, gzyx: get_hansen_vector_obs(ayx, grid, None, base_n)
        else:  # No goal
            space = Discrete(int(2 ** base_n * (base_n + 1)))
            obs = lambda ayx, gyx: get_hansen_obs(ayx, grid, gyx, base_n)
    elif 'grid' in obs_type: raise NotImplementedError
    else: raise NotImplementedError('Observation type not recognized')
    return space, obs


def compute_obs_space(cnst_layout: np.ndarray, obs_n: int = 0) -> Union[Box, Discrete]:
    """Compute observation space

    Discrete observation space has one state for each possible position of agent. Should I include goal?
    Hansen observation space has 4 (N, S, E, W) ** 4 (empty, wall, stair, goal) possible states. Probably actually fewer possible states, but eh
    Grid observation space is a (obs_n, obs_n) uint8 box space with aliased stairs

    Args:
        cnst_layout: Full map (z, y, x)
        obs_n: 0 is discrete, 1 is hansen, 3+ is surround box, even numbers are invalid
    Returns:
        space: gym observation space
    """
    if not obs_n:
        return Discrete((cnst_layout != GR_CNST.wall).sum())
    elif obs_n == 1:
        return Discrete(4**4)
    assert (obs_n % 2) == 1
    return Box(0, 4, shape=(obs_n, obs_n), dtype=int)