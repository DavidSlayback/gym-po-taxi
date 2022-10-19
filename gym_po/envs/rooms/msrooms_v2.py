from typing import Tuple, Sequence, Union, Callable, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from dotsi import DotsiDict
import gymnasium
from gymnasium.core import ObsType, ActType
from gymnasium.spaces import Discrete, Box, Space
from gymnasium.vector.utils import batch_space
from .render_utils import *
from .observations import get_number_discrete_states_and_conversion
from .actions import *


# Fixed locations
END_XYZ = (9, 7, -1)  # East hallway
START_XYZ = (1, 1, 0)  # NW Cornergy
SW = (11, 1); SW_NP = np.array(SW)  # Downstairs
NE = (1, 11); NE_NP = np.array(NE)  # Upstairs
upstairs = NE = np.array([1, 11])
downstairs = SW = np.array([11, 1])

# Constant integers for each object
GR_CNST = DotsiDict({
    'wall': 0,
    'goal': 1,
    'stair_down': 2,
    'stair_up': 3,
})
MAX_GR_CNST = len(GR_CNST) - 1
# Constant colors, can be indexed by values above
GR_CNST_COLORS = DotsiDict({
    'wall': COLORS.black,
    'empty': COLORS.gray_dark,
    'agent': COLORS.green,
    'goal': COLORS.blue,
    'stair_up': COLORS.gray_light,
    'stair_down': COLORS.gray,
})


# 13x13 FourRooms, upstairs and downstairs locations marked by "U" and "D"
FR_MAP = np.array(
    [
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, 3, 3, 3, 3, 3, -1, 0, 0, 0, 0, 0, -1],
        [-1, 3, 3, 3, 3, 3, -1, 0, 0, 0, 0, 0, -1],
        [-1, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, -1],
        [-1, 3, 3, 3, 3, 3, -1, 0, 0, 0, 0, 0, -1],
        [-1, 3, 3, 3, 3, 3, -1, 0, 0, 0, 0, 0, -1],
        [-1, -1, 2, -1, -1, -1, -1, 0, 0, 0, 0, 0, -1],
        [-1, 2, 2, 2, 2, 2, -1, -1, -1, 0, -1, -1, -1],
        [-1, 2, 2, 2, 2, 2, -1, 1, 1, 1, 1, 1, -1],
        [-1, 2, 2, 2, 2, 2, -1, 1, 1, 1, 1, 1, -1],
        [-1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, -1],
        [-1, 2, 2, 2, 2, 2, -1, 1, 1, 1, 1, 1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    ]
)

def rooms_map_to_multistory(map: NDArray[int] = FR_MAP, num_floors: int = 1) -> Tuple[NDArray[int], NDArray[int]]:
    """Convert numbered room map into multistory walking and room map

    Args:
        map: Room map (e.g., FR_MAP)
        num_floors: Number of floors

    Returns:
        walk_map: Multistory walking map
        room_map: Multistory rooms map (room numbers increase based on floors)
    """
    walk_map = map.copy()
    walk_map[map >= 0] = 0  # Alias the rooms for this layout
    walk_map += 1  # Walls are 0s, everything else is 1
    ms = np.stack([walk_map for _ in range(num_floors)], 0)
    n_rooms = map.max()  # can work with different numbers of rooms
    ms_rooms = np.stack([map[map >= 0] + i * n_rooms for i in range(num_floors)], 0)
    if num_floors > 1:
        ms[1:, downstairs[0], downstairs[1]] = GR_CNST.stair_down
        ms[:-1, upstairs[0], upstairs[1]] = GR_CNST.stair_up
    return ms, ms_rooms


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
    first_val = MAX_GR_CNST + 1
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


def get_hansen_vector_obs(agent_zyx: np.ndarray, grid: np.ndarray, goal_zyx: Optional[np.ndarray] = None, hansen_n: int = 8) -> np.ndarray:
    """Same as above, but a vector representation (like the grid obs, but flattened)

    Args:
        agent_zyx: (z, y,x) coordinate of agent(s) [B,3]
        grid: (z, y,x) numpy grid
        goal_zyx (z, y,x) goal location(s) [B, 3]
        hansen_n: 8 or 4
    Returns:
        Obs (Constants)
    """
    a = ACTIONS_CARDINAL_Z if hansen_n == 4 else ACTIONS_ORDINAL_Z  # Observation only on agent floor
    a = a[None, :]
    coords = agent_zyx[:, None] + a
    squares = grid[tuple(coords.transpose(2, 0, 1))]
    # So each square can be wall, empty, stair, goal
    squares[(squares > 0) & (squares <= MAX_GR_CNST)] = 2  # Alias stairs (2)
    squares[squares > MAX_GR_CNST] = 1  # Rooms all become same "empty" squares (1)
    if goal_zyx is not None:
        is_goal = (goal_zyx[:, None] == coords).all(-1)
        squares[is_goal] = 3  # Add goal
    return squares


def get_hansen_obs(agent_zyx: np.ndarray, ms_grid: np.ndarray, goal_zyx: np.ndarray, hansen_n: int = 8) -> int:
    """Get hansen observation of agent(s) (empty, wall), goal in (null, N, E, S, W) based on grid

    Args:
        agent_zyx: (y, x) coordinate of agent(s) [B, 2]
        ms_grid: (y, x) numpy grid
        goal_zyx: (y, x) goal location(s)
        hansen_n: 8 or 4
    Returns:
        obs
    """
    a = ACTIONS_CARDINAL_Z if hansen_n == 4 else ACTIONS_ORDINAL_Z
    a = a[None, :]
    coords = agent_zyx[:, None] + a
    # is_goal = (goal_yx[:, None] == coords).all(-1)
    where_is_goal = np.nonzero((goal_zyx[:, None] == coords).all(-1))
    goal_mult = np.ones(goal_zyx.shape[0])
    goal_mult[where_is_goal[0]] = where_is_goal[1] + 1
    squares = ms_grid[tuple(coords.transpose(2, 0, 1))]
    # So each square can be wall, empty, stair. Goal added separately
    squares[(squares > 0) & (squares <= MAX_GR_CNST)] = 2  # Alias stairs (2)
    squares[squares > MAX_GR_CNST] = 1  # Rooms all become same "empty" squares (1)
    multipliers = np.array([3 ** i for i in range(a.shape[1])])  # There's only one goal, let's multiply it separately after
    return squares.dot(multipliers) * goal_mult


def get_observation_space_and_function(obs_type: str, ms_grid: np.ndarray, obs_n: int = 3) -> Tuple[Space, Callable[[np.ndarray, np.ndarray], np.ndarray]]:
    """Return Gym Space and observation function (takes agent and goal positions (zyx) as input)"""
    is_vector = 'vector' in obs_type  # Return a vector or a scalar? In practice, scalar can get huge
    has_goal = 'goal' in obs_type  # Should goal information be included?
    a_max = np.array(ms_grid.shape) - 2; a_max[0] += 1  # Max agent coordinate in 3-D
    a_min = np.array([0, 1, 1])  # Min agent coordinate in 3-D
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
                space = Box(np.tile(a_min, 2), np.tile(a_max, 2), (6,), dtype=int)
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
                space = Box(0, 3, (base_n,), dtype=int)
                obs = lambda azyx, gzyx: get_hansen_vector_obs(azyx, ms_grid, gzyx, base_n)
            else:
                space = Box(0, 2, (base_n,), dtype=int)
                obs = lambda azyx, gzyx: get_hansen_vector_obs(azyx, ms_grid, None, base_n)
        else:  # Goal
            space = Discrete(int(3 ** base_n * (base_n + 1)))
            obs = lambda azyx, gzyx: get_hansen_obs(azyx, ms_grid, gzyx, base_n)
    elif 'grid' in obs_type: raise NotImplementedError
    else: raise NotImplementedError('Observation type not recognized')
    return space, obs

class MultistoryFourRoomsEnvV2(gymnasium.Env):
    """Vectorized Multistory FourRooms environment, using tricks from ROOMS/CROOMS"""
    metadata = {"name": "MultistoryFourRoomsV2", "render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}

    def __init__(self, num_envs: int, grid_z: int = 1, floor_map: str = BASE_FOURROOMS_MAP_WITH_STAIRS, time_limit: int = 500,
                 obs_type: str = 'mdp', obs_n: int = 3, action_failure_probability: float = (1./ 3), action_type: str = 'cardinal',
                 agent_xyz: Optional[Sequence[int]] = None, goal_xyz: Optional[Sequence[int]] = (0, 0, 0),
                 step_reward: float = 0., wall_reward: float = 0., goal_reward: float = 1., render_mode: Optional[str] = None,
                 **kwargs):
        """

        :param num_envs: Number of environments in parallel
        :param grid_z: Number of floors
        :param floor_map: Floor map string to use as base
        :param time_limit: Max time before episode terminates
        :param obs_type: Type of observation. One of 'discrete', 'hansen', 'hansen8', 'vector_hansen', 'vector_hansen8', 'room', 'grid'
                hansen is 4 adjacent <empty|wall|goal>, hansen8 is 8. room treats each room as an obs
        :param obs_n: If 'grid' observation, use NxN grid centered on agent
        :param action_failure_probability: Likelihood that taking one action fails and chooses another
        :param action_type: 'ordinal' (8D compass) or 'cardinal' (4D compass)
        :param agent_xyz: Optionally, provide fixed agent location. If None, random. If 2D, bottom floor. If invalid, use default fixed start
        :param goal_xyz: Optionally, provide fixed goal location. As above
        :param step_reward: Reward for each step
        :param wall_reward: Reward for hitting a wall
        :param goal_reward: Reward for reaching goal
        :param kwargs:  Throw these away
        """
        self.grid, self.img = generate_layout_and_img(floor_map, grid_z)
        self.metadata['name'] += f'{grid_z}__{action_type}__{obs_type}'
        self.gridshape = np.array(self.grid.shape)
        self.single_observation_space, self._get_obs = get_observation_space_and_function(obs_type, self.grid, obs_n)
        spawn_vs = np.array(np.nonzero(self.grid > MAX_GR_CNST))  # [3, N]
        self.valid_states = np.flatnonzero(self.grid > 0)
        self.valid_agent_states = np.ravel_multi_index(spawn_vs[:, spawn_vs[0] == 0], self.grid.shape)
        self.valid_goal_states = np.ravel_multi_index(spawn_vs[:, spawn_vs[0] == self.gridshape[0] - 1], self.grid.shape)
        self.render_mode = render_mode

        self.actions = ACTIONS_CARDINAL_Z if action_type == 'cardinal' else ACTIONS_ORDINAL_Z
        # Boilerplate for vector environment
        self.num_envs = num_envs
        self.is_vector_env = True
        self.single_action_space = Discrete(self.actions.shape[0])
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.observation_space = batch_space(self.single_observation_space, num_envs)

        # Constants
        self.time_limit = time_limit
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.wall_reward = wall_reward

        # Random or fixed goal/agent
        if goal_xyz is not None:
            goal_zyx = tuple(reversed(goal_xyz))  # (x,y) to (y,x)
            if self.grid[goal_zyx] <= MAX_GR_CNST: goal_zyx = tuple(reversed(END_XYZ))  # Goal can't be on stairs
            goal_zyx = np.array(goal_zyx)
            if goal_zyx[0] == -1: goal_zyx[0] = self.gridshape[0] - 1
            self._sample_goal = lambda b, rng: np.full((b, 3), goal_zyx, dtype=int)
        else: self._sample_goal = lambda b, rng: np.array(np.unravel_index(rng.choice(self.valid_goal_states, b), self.grid.shape)).swapaxes(0,1)
        if agent_xyz is not None:
            agent_zyx = tuple(reversed(agent_xyz))
            agent_zyx = np.array(agent_zyx)
            if self.grid[agent_zyx] == GR_CNST.wall: agent_zyx = tuple(reversed(START_XYZ))
            self._sample_agent = lambda b, rng: np.full((b, 3), agent_zyx, dtype=int)
        else: self._sample_agent = lambda b, rng: np.array(np.unravel_index(rng.choice(self.valid_agent_states, b), self.grid.shape)).swapaxes(0,1)
        self.action_matrix = create_action_probability_matrix(self.actions.shape[0], action_failure_probability)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Union[ObsType, tuple[ObsType, dict]]:
        """Reset all environments, set seed if given"""
        super().reset(seed, options)
        self.elapsed = np.zeros(self.num_envs, int)
        self.goal_zyx = self._sample_goal(self.num_envs, self.np_random)
        self.agent_zyx = self._sample_agent(self.num_envs, self.np_random)
        obs = self._get_obs(self.agent_zyx, self.goal_zyx)
        return obs, {}

    def _reset_some(self, mask: np.ndarray):
        """Reset only a subset of environments"""
        if b := mask.sum():
            self.elapsed[mask] = 0
            self.goal_zyx[mask] = self._sample_goal(b, self.np_random)
            self.agent_zyx[mask] = self._sample_agent(b, self.np_random)

    def step(self, action: ActType) -> Tuple[ObsType, np.ndarray, np.ndarray, np.ndarray, Union[dict, list]]:
        """Step in environment

        Sample random action failure. Move agent(s) where move is valid.
        Check if we reached goal. Update with step, wall, and goal rewards.
        """
        self.elapsed += 1
        # Movement
        a = vectorized_multinomial_with_rng(self.action_matrix[action], self.np_random)
        proposed_zyx = self.agent_zyx + self.actions[a]
        oob = self._out_of_bounds(proposed_zyx)
        self.agent_zyx[~oob] = proposed_zyx[~oob]
        self._transit_stairs(~oob)
        # Reward
        r = np.zeros(self.num_envs, dtype=np.float32)
        d = (self.agent_zyx == self.goal_zyx).all(-1)
        r += self.step_reward
        r[oob] = self.wall_reward
        r[d] = self.goal_reward
        truncated = self.elapsed > self.time_limit
        self._reset_some(d | truncated)
        return self._get_obs(self.agent_zyx, self.goal_zyx), r, d, truncated, {}

    def _out_of_bounds(self, proposed_zyx: np.ndarray) -> np.ndarray:
        """Return whether given coordinates correspond to empty/goal square.

        Rooms are surrounded by walls, so only need to check this"""
        # oob = (proposed_yx >= self.gridshape[None, :]).any(-1) | (proposed_yx < 0).any(-1)
        # oob[~oob] = self.grid[tuple(proposed_yx[~oob].T)] == -1
        # return oob
        return self.grid[tuple(proposed_zyx.T)] == GR_CNST.wall


    def _transit_stairs(self, moved: np.ndarray):
        """If we MOVED (not oob) and we're on stairs, transit them"""
        go_up = (self.grid[tuple(self.agent_zyx.T)] == GR_CNST.stair_up) & moved
        go_down = (self.grid[tuple(self.agent_zyx.T)] == GR_CNST.stair_down) & moved
        if go_up.any():
            self.agent_zyx[go_up, 0] += 1
            self.agent_zyx[go_up, 1:] = SW_NP
        if go_down.any():
            self.agent_zyx[go_down, 0] -= 1
            self.agent_zyx[go_down, 1:] = NE_NP


    # def render(self, mode="human", idx: Optional[Sequence[int]] = None):
    #     """Render environment as an rgb array, with highlighting of agent's view. If "human", render with pygame"""
    #     if idx is None: idx = np.arange(1)
    #     idx = np.array(idx)
    #     # zs = np.zeros_like(idx)
    #     a, g = self.agent[idx], self.goal[idx]
    #     ag, gg = self._to_grid(a), self._to_grid(g)  # Grid coordinates
    #     img = self.img[ag[0]].copy()  # Get agent's floor
    #     img[(idx,) + tuple(ag)[1:]] = GR_CNST_COLORS.agent  # Add agent (always same floor)
    #     goal_on_agent_floor = (gg[0] == ag[0])
    #     img[(idx[goal_on_agent_floor],) + tuple(gg[:, goal_on_agent_floor])[1:]] = GR_CNST_COLORS.goal  # Add goal if on same floor
    #     v = np.concatenate((idx[None, :], ag[1:]), axis=0)
    #     if self.obs_n == 1: # Hansen, render floor, highlight hansen grid
    #         v = (ag[1:, None] + self.ACTIONS[1:][...,None]).reshape(2, -1)
    #         idx = np.tile(idx, (1,4))
    #         v = np.concatenate((idx, v), axis=0)
    #     elif self.obs_n > 1:
    #         v = self._to_grid(self._last_grid_obs_coords[idx]) # Use cached coordinates
    #         v[0, idx] = idx[:,None,None]
    #     img[tuple(v)] += 40  # lighten
    #     img = tile_images(img)  # Tile
    #     img = resize(img, CELL_PX)
    #     if mode in ('rgb_array', 'rgb'): return img
    #     else:
    #         import pygame
    #         if self._viewer is None:
    #             pygame.init()
    #             self._viewer = pygame.display.set_mode(img.shape[:-1])
    #         sfc = pygame.surfarray.make_surface(img.swapaxes(0, 1))
    #         self._viewer.blit(sfc, (0, 0))
    #         pygame.display.update()
    #         return img



