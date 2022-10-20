from enum import IntEnum
from typing import Tuple, Sequence, Union, Callable, Optional, List

import gymnasium
import numpy as np
from dotsi import DotsiDict
from gymnasium.core import ObsType, ActType, RenderFrame
from gymnasium.spaces import Discrete, Box, Space
from gymnasium.vector.utils import batch_space
from numpy.typing import NDArray

from .action_utils import *
from .observations import get_number_discrete_states_and_conversion
from .render_utils import *

# Fixed locations
END_XYZ = (9, 7, -1)  # East hallway
START_XYZ = (1, 1, 0)  # NW Cornergy
SW = (11, 1)
SW_NP = np.array(SW)  # Downstairs
NE = (1, 11)
NE_NP = np.array(NE)  # Upstairs
upstairs = NE = np.array([1, 11])
downstairs = SW = np.array([11, 1])

# Constant integers for each object
class GR_CNST(IntEnum):
    wall = 0
    goal = 1
    stair_down = 2
    stair_up = 3


MAX_GR_CNST = int(max(GR_CNST))

# Constant colors, can be indexed by values above
GR_CNST_COLORS = DotsiDict(
    {
        "wall": COLORS.black,
        "empty": COLORS.gray_dark,
        "agent": COLORS.green,
        "goal": COLORS.blue,
        "stair_up": COLORS.gray_light,
        "stair_down": COLORS.gray,
    }
)


# 13x13 FourRooms, walls are 0, rooms are 1-4 (clockwise)
FR_MAP = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 4, 4, 4, 4, 4, 0, 1, 1, 1, 1, 1, 0],
        [0, 4, 4, 4, 4, 4, 0, 1, 1, 1, 1, 1, 0],
        [0, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 0],
        [0, 4, 4, 4, 4, 4, 0, 1, 1, 1, 1, 1, 0],
        [0, 4, 4, 4, 4, 4, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 3, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
        [0, 3, 3, 3, 3, 3, 0, 0, 0, 1, 0, 0, 0],
        [0, 3, 3, 3, 3, 3, 0, 2, 2, 2, 2, 2, 0],
        [0, 3, 3, 3, 3, 3, 0, 2, 2, 2, 2, 2, 0],
        [0, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 0],
        [0, 3, 3, 3, 3, 3, 0, 2, 2, 2, 2, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)


def rooms_map_to_multistory(
    map: NDArray[int] = FR_MAP, num_floors: int = 1
) -> Tuple[NDArray[int], NDArray[int]]:
    """Convert numbered room map into multistory walking and room map

    Args:
        map: Room map (e.g., FR_MAP) [NxN]
        num_floors: Number of floors [S]

    Returns:
        walk_map: Multistory walking map with stairs [SxNxN]
        room_map: Multistory rooms map (room numbers increase based on floors) [SxNxN]
    """
    walk_map = map.copy()
    walk_map[map > 0] = 1  # Alias the rooms for this layout
    ms = np.stack([walk_map for _ in range(num_floors)], 0)
    n_rooms = map.max() - 1  # can work with different numbers of rooms
    ms_rooms = np.stack([map[map > 0] + i * n_rooms for i in range(num_floors)], 0)
    if num_floors > 1:
        ms[1:, downstairs[0], downstairs[1]] = GR_CNST.stair_down
        ms[:-1, upstairs[0], upstairs[1]] = GR_CNST.stair_up
    return ms, ms_rooms


def walk_map_to_state_map(walk_map: NDArray[int]) -> Tuple[NDArray[int], int]:
    """Convert walk map (wall, goal, stairs) to state map (1 state per valid agent square)

    Args:
        walk_map: Multistory walking map with stairs [SxNxN]

    Returns:
        state_map: Coord to discrete state grid [SxNxN]
        n_states: Number of unique agent state positions

    """
    state_map = walk_map[walk_map >= GR_CNST.wall].reshape(walk_map.shape)
    n_states = state_map.max() - 1
    return state_map, n_states


def generate_layouts_and_img(
    map: NDArray[int] = FR_MAP, grid_z: int = 1
) -> Tuple[NDArray[int], NDArray[int], NDArray[int]]:
    """Generate walk map (GR_CNST), room_map, and image from base fourrooms map

    Args:
        map: Base fourrooms map [NxN]
        grid_z: Number of floors [S]
    Returns:
        walk_map: Multistory walking map with stairs [SxNxN]
        room_map: Multistory rooms map (room numbers increase based on floors) [SxNxN]
        img_map: Multistory image map (filled in with colors, just needs resize) [SxNxNx3]
    """
    walk_map, room_map = rooms_map_to_multistory(map, grid_z)
    img_map = np.zeros_like(
        walk_map, shape=(*walk_map.shape, 3), dtype=np.uint8
    )  # Unscaled image version
    for (k, v) in GR_CNST.__members__.items():
        img_map[walk_map == v] = GR_CNST_COLORS[k]
    return walk_map, room_map, img_map


def get_hansen_vector_obs(
    agent_zyxNDArray,
    gridNDArray,
    goal_zyx: Optional[np.ndarray] = None,
    hansen_n: int = 8,
) -> np.ndarray:
    """Same as above, but a vector representation (like the grid obs, but flattened)

    Args:
        agent_zyx: (z, y,x) coordinate of agent(s) [B,3]
        grid: (z, y,x) numpy grid
        goal_zyx (z, y,x) goal location(s) [B, 3]
        hansen_n: 8 or 4
    Returns:
        Obs (Constants)
    """
    a = (
        ACTIONS_CARDINAL_Z if hansen_n == 4 else ACTIONS_ORDINAL_Z
    )  # Observation only on agent floor
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


def get_hansen_obs(
    agent_zyxNDArray, ms_gridNDArray, goal_zyxNDArray, hansen_n: int = 8
) -> int:
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
    multipliers = np.array(
        [3**i for i in range(a.shape[1])]
    )  # There's only one goal, let's multiply it separately after
    return squares.dot(multipliers) * goal_mult


def get_observation_space_and_function(
    obs_type: str, ms_gridNDArray, obs_n: int = 3
) -> Tuple[Space, Callable[[np.ndarray, np.ndarray], np.ndarray]]:
    """Return Gym Space and observation function (takes agent and goal positions (zyx) as input)"""
    is_vector = (
        "vector" in obs_type
    )  # Return a vector or a scalar? In practice, scalar can get huge
    has_goal = "goal" in obs_type  # Should goal information be included?
    a_max = np.array(ms_grid.shape) - 2
    a_max[0] += 1  # Max agent coordinate in 3-D
    a_min = np.array([0, 1, 1])  # Min agent coordinate in 3-D
    if "room" in obs_type:
        assert not is_vector
        offset = len(GR_CNST)
        n = (
            ms_grid.max() - offset
        )  # Number of distinct rooms, offset by number of other state types
        if has_goal:
            space = Discrete(int(n**2))
            obs = lambda azyx, gzyx: (ms_grid[tuple(azyx.T)] - offset) + n * (
                ms_grid[tuple(gzyx.T)] - offset
            )
        else:
            space = Discrete(int(n))
            obs = lambda azyx, gzyx: ms_grid[tuple(azyx.T)]
    elif "mdp" in obs_type:  # Fully observable (if goal is fixed or provided)
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
                space = Discrete(int(n**2))
                obs = (
                    lambda azyx, gzyx: state_grid[tuple(azyx.T)]
                    + n * state_grid[tuple(gzyx.T)]
                )
            else:
                space = Discrete(int(n))
                obs = lambda azyx, gzyx: state_grid[tuple(azyx.T)]
    elif "hansen" in obs_type:
        base_n = 8 if "8" in obs_type else 4
        if is_vector:
            if has_goal:
                space = Box(0, 3, (base_n,), dtype=int)
                obs = lambda azyx, gzyx: get_hansen_vector_obs(
                    azyx, ms_grid, gzyx, base_n
                )
            else:
                space = Box(0, 2, (base_n,), dtype=int)
                obs = lambda azyx, gzyx: get_hansen_vector_obs(
                    azyx, ms_grid, None, base_n
                )
        else:  # Goal
            space = Discrete(int(3**base_n * (base_n + 1)))
            obs = lambda azyx, gzyx: get_hansen_obs(azyx, ms_grid, gzyx, base_n)
    else:
        raise NotImplementedError("Observation type not recognized")
    return space, obs


class MultistoryFourRoomsEnv(gymnasium.Env):
    """Vectorized Multistory FourRooms environment, using tricks from ROOMS/CROOMS"""

    metadata = {
        "name": "MultistoryFourRoomsV2",
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(
        self,
        num_envs: int,
        grid_z: int = 1,
        floor_map: NDArray = FR_MAP,
        time_limit: int = 500,
        obs_type: str = "mdp",
        obs_n: int = 3,
        action_failure_probability: float = 1.0 / 3,
        action_type: str = "cardinal",
        agent_xyz: Optional[Sequence[int]] = None,
        goal_xyz: Optional[Sequence[int]] = END_XYZ,
        step_reward: float = 0.0,
        wall_reward: float = 0.0,
        goal_reward: float = 1.0,
        render_mode: Optional[str] = None,
        **kwargs,
    ):
        """Create an instance of Multistory FourRooms

        Args:
            num_envs: Number of parallel environments N
            grid_z: Number of stories S
            floor_map: Base map for each story
            time_limit: Max timesteps per episode
            obs_type: Type of observation. Keywords are 'vector' (vector instead of discrete),
                'goal' (include goal in observation),
                'mdp' (use agent position)
                'hansen' (use adjacent, include '8' for use ordinal instead of cardinal)
            action_failure_probability: Probability of taking random action instead of selected
            action_type: 'cardinal' (NSEW) or 'ordinal' (include NE-SE-SW-NW)
            agent_xyz: If provided, fixed agent spawn position. Otherwise randomly spawn bottom floor
            goal_xyz: If provided, fixed goal spawn position. Otherwise random spawn top floor
            step_reward: Reward per step
            wall_reward: Reward for hitting a wall
            goal_reward: Reward for reaching goal
            render_mode: Type of rendering
            **kwargs: Ignored
        """
        self.grid, self.room_grid, self.img = generate_layouts_and_img(
            floor_map, grid_z
        )
        self.metadata["name"] += f"{grid_z}__{action_type}__{obs_type}"
        self.gridshape = np.array(self.grid.shape)
        (
            self.single_observation_space,
            self._get_obs,
        ) = get_observation_space_and_function(obs_type, self.grid, obs_n)
        spawn_vs = np.array(np.nonzero(self.grid > GR_CNST.wall))  # [3, N]
        self.valid_states = np.flatnonzero(self.grid > GR_CNST.wall)
        self.valid_agent_states = np.ravel_multi_index(
            spawn_vs[:, spawn_vs[0] == 0], self.grid.shape
        )
        self.valid_goal_states = np.ravel_multi_index(
            spawn_vs[:, spawn_vs[0] == self.gridshape[0] - 1], self.grid.shape
        )
        self.render_mode = render_mode

        self.actions = (
            ACTIONS_CARDINAL_Z if action_type == "cardinal" else ACTIONS_ORDINAL_Z
        )
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
            if self.grid[goal_zyx] <= MAX_GR_CNST:
                goal_zyx = tuple(reversed(END_XYZ))  # Goal can't be on stairs
            goal_zyx = np.array(goal_zyx)
            if goal_zyx[0] == -1:
                goal_zyx[0] = self.gridshape[0] - 1
            self._sample_goal = lambda b, rng: np.full((b, 3), goal_zyx, dtype=int)
        else:
            self._sample_goal = lambda b, rng: np.array(
                np.unravel_index(rng.choice(self.valid_goal_states, b), self.grid.shape)
            ).swapaxes(0, 1)
        if agent_xyz is not None:
            agent_zyx = tuple(reversed(agent_xyz))
            agent_zyx = np.array(agent_zyx)
            if self.grid[agent_zyx] == GR_CNST.wall:
                agent_zyx = tuple(reversed(START_XYZ))
            self._sample_agent = lambda b, rng: np.full((b, 3), agent_zyx, dtype=int)
        else:
            self._sample_agent = lambda b, rng: np.array(
                np.unravel_index(
                    rng.choice(self.valid_agent_states, b), self.grid.shape
                )
            ).swapaxes(0, 1)
        self.action_matrix = create_action_probability_matrix(
            self.actions.shape[0], action_failure_probability
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Union[ObsType, tuple[ObsType, dict]]:
        """Reset all environments, set seed if given"""
        super().reset(seed=seed, options=options)
        self.elapsed = np.zeros(self.num_envs, int)
        self.goal_zyx = self._sample_goal(self.num_envs, self.np_random)
        self.agent_zyx = self._sample_agent(self.num_envs, self.np_random)
        obs = self._get_obs(self.agent_zyx, self.goal_zyx)
        return obs, {}

    def _reset_some(self, maskNDArray):
        """Reset only a subset of environments"""
        if b := mask.sum():
            self.elapsed[mask] = 0
            self.goal_zyx[mask] = self._sample_goal(b, self.np_random)
            self.agent_zyx[mask] = self._sample_agent(b, self.np_random)

    def step(
        self, action: ActType
    ) -> Tuple[ObsType, np.ndarray, np.ndarray, np.ndarray, Union[dict, list]]:
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

    def _out_of_bounds(self, proposed_zyxNDArray) -> np.ndarray:
        """Return whether given coordinates correspond to empty/goal square"""
        return self.grid[tuple(proposed_zyx.T)] == GR_CNST.wall

    def _transit_stairs(self, movedNDArray):
        """If we MOVED (not oob) and we're on stairs, transit them"""
        go_up = (self.grid[tuple(self.agent_zyx.T)] == GR_CNST.stair_up) & moved
        go_down = (self.grid[tuple(self.agent_zyx.T)] == GR_CNST.stair_down) & moved
        if go_up.any():
            self.agent_zyx[go_up, 0] += 1
            self.agent_zyx[go_up, 1:] = SW_NP
        if go_down.any():
            self.agent_zyx[go_down, 0] -= 1
            self.agent_zyx[go_down, 1:] = NE_NP

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        """Rendering, ignored for now"""
        raise NotImplementedError
