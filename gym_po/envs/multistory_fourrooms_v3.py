from typing import Sequence, Tuple, Callable, Optional, NamedTuple, Union
import numpy as np
import gym
from gym.spaces import Box, Discrete
from gym.vector.utils import batch_space
from gym.utils.seeding import np_random
from dotsi import DotsiDict

from .grid_utils import WALLS, DIRECTIONS_3D_NP, get_coord_to_flat_function, get_flat_to_coord_function
from .render_utils import COLORS, resize, draw_text_at, CELL_PX
from .action_utils import generate_action_probability_matrix, vectorized_multinomial_with_rng

# Fourrooms minus the external wall (13x13 -> 11x11)
# Upstairs and downstairs locations marked by "U" and "D"
BASE_FOURROOMS_MAP_WITH_STAIRS = (
    "     |    U",
    "     |     ",
    "           ",
    "     |     ",
    "     |     ",
    "- ---|     ",
    "     |-- --",
    "     |     ",
    "     |     ",
    "           ",
    "D    |     ",
)


# Constant integers for each object
GR_CNST = DotsiDict({
    'wall': 0,
    'empty': 1,
    'agent': 2,
    'goal': 3,
    'stair_up': 4,
    'stair_down': 5
})
# Constant colors, can be indexed by values above
GR_CNST_COLORS = DotsiDict({
    'wall': COLORS.black,
    'empty': COLORS.white,
    'agent': COLORS.green,
    'goal': COLORS.blue,
    'stair_up': COLORS.gray_light,
    'stair_down': COLORS.gray_dark
})

HANSEN_MULTIPLIERS = np.array([1, 4, 16, 64])  # 4 possibilities for adjacent observations

def convert_str_map_to_walled_np_cnst(map: Sequence[str]) -> np.ndarray:
    """Return a bordered version of the map converted to grid constants (for image and navigation)"""
    str_map = np.pad(np.asarray(map, dtype='c').astype(str), 1, constant_values='|')  # Border
    cnst_map = np.zeros_like(str_map, dtype=np.uint8)  # Constant "wall"
    cnst_map[str_map == ' '] = GR_CNST.empty  # Fill in empty
    cnst_map[str_map == 'U'] = GR_CNST.stair_up
    cnst_map[str_map == 'D'] = GR_CNST.stair_down
    return cnst_map


def generate_layout_and_img(map: Sequence[str], grid_z: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Return full (z,y,x) map layout and unscaled (z,y,x,3) rgb image

    Args:
        map: Raw string map
        grid_z: Number of floors
    Returns:
        cnst_layout: (z,y,x) grid layout using constants
        img_map (z,y,x,3) unscaled img
    """
    cnst_map = convert_str_map_to_walled_np_cnst(map)  # Get bordered constant version (y,x)
    cnst_layout = np.stack(tuple(cnst_map for _ in range(grid_z)), axis=0)
    if grid_z == 1:  # Strip out stairs
        cnst_layout[cnst_layout != GR_CNST.wall] = GR_CNST.empty
    else:  # Remove downstairs from bottom, upstairs from top
        cnst_layout[0, cnst_layout[0] == GR_CNST.stair_down] = GR_CNST.empty
        cnst_layout[-1, cnst_layout[-1] == GR_CNST.stair_up] = GR_CNST.empty
    img_map = np.zeros_like(cnst_layout, shape=(*cnst_layout.shape, 3))  # Unscaled image version
    for (k, v) in GR_CNST.items():
        img_map[cnst_layout == v] = GR_CNST_COLORS[k]
    return cnst_layout, img_map


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
    return Box(0, 4, shape=(obs_n, obs_n), dtype=np.uint8)


class MultistoryFourRoomsVecEnv(gym.Env):
    """Vectorized Multistory fourrooms environment"""
    metadata = {"name": "MultistoryFourRooms", "render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    ACTIONS = DIRECTIONS_3D_NP[:, :4]
    ACTION_NAMES = ['Up', 'Down', 'Left', 'Right']

    def __init__(self, num_envs: int, grid_z: int = 1, map: Sequence[str] = BASE_FOURROOMS_MAP_WITH_STAIRS,
                 seed: Optional[int] = None, time_limit: int = 100, obs_n: int = 1,
                 action_failure_probability: float = 1./3, agent_floor: int = 0, goal_floor: int = -1,
                 agent_location: Optional[Tuple[int, int]] = None, goal_location: Optional[Tuple[int, int]] = None,
                 wall_reward: float = 0.):
        """Create a multistory four rooms environment

        Args:
            num_envs: Number of vectorized environments
            grid_z: Number of floors
            map: Base (unbordered) map for floors. 'U' is upstairs, 'D' is downstairs
            seed: Seed to use
            time_limit: Max number of timesteps before an environment resets
            obs_n: If 0, discrete, if 1, use hansen, otherwise use grid observations
            action_failure_probability: Probability that a chosen action will fail
            agent_floor: Floor where agent can spawn (defaults to bottom, negative below -1 means all floors)
            goal_floor: Floor where goal can spawn (defaults to top, negative below -1 means all floors)
            agent_location: If provided, a fixed (yx) location where agent spawns each episode (defaults random)
            goal_location: If provided, a fixed (yx) location where goal spawns each episode (defaults to east hallway)
            wall_reward: Reward for hitting a wall, should be negative (defaults to 0
        """
        # VectorEnv
        self.num_envs = num_envs
        self.is_vector_env = True
        # Time limit
        self.time_limit = time_limit
        self.elapsed = np.zeros(num_envs, dtype=int)
        # Rewards
        self.wall_reward = wall_reward
        # Action space
        self.single_action_space = Discrete(len(self.ACTION_NAMES))
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.action_probability_matrix = generate_action_probability_matrix(action_failure_probability, self.single_action_space.n)
        # Observation space
        self.grid, self.img = generate_layout_and_img(map, grid_z)
        self.flat_grid = self.grid.ravel(order='C')
        self._shape = self.grid.shape
        self.single_observation_space = compute_obs_space(self.grid, obs_n)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        ec = (self.flat_grid != GR_CNST.wall).nonzero()[0]
        self.f_coord_to_e_coord = np.isin(np.arange(self.grid.size), ec, True).cumsum() - 1  # Convert agent coordinates to discrete observations (of which there are fewer)
        self.set_obs_fn(obs_n)
        # Grid-to-flat and flat-to-grid conversions (assume 'C' order)
        self._to_flat = get_coord_to_flat_function(self._shape)  # (ncoord,)
        self._to_grid = get_flat_to_coord_function(self._shape)  # (ndim, ncoord)
        # Flat actions
        f_ups, f_downs = 0,0
        if grid_z > 1:
            ne, sw = (self.flat_grid == GR_CNST.stair_up).nonzero()[0], (self.flat_grid == GR_CNST.stair_down).nonzero()[0]
            f_ups, f_downs = sw[0] - ne[0], ne[0] - sw[0]
        base_coord = np.ones((3,), dtype=int)
        fbc = self._to_flat(base_coord)
        fbc_act = self._to_flat(base_coord[...,None] + self.ACTIONS)
        self.flat_actions = fbc_act - fbc
        self.flat_actions = np.concatenate((self.flat_actions, np.array([f_ups, f_downs])), axis=0)
        # Agent and goal sampling
        self.agent_floor = np.array(agent_floor) if agent_floor >= -1 else np.arange(self._shape[0])
        self.agent_floor[self.agent_floor == -1] = self.grid.shape[0] - 1
        self.goal_floor = np.array(goal_floor) if goal_floor >= -1 else np.arange(self._shape[0])
        self.goal_floor[self.goal_floor == -1] = self.grid.shape[0] - 1
        vs = np.array((self.grid == GR_CNST.empty).nonzero())  # Anywhere an agent or goal can spawn
        self.agent_spawn_locations = self._to_flat(vs[:, np.isin(vs[0], self.agent_floor)]) if agent_location is None else self._to_flat(vs[:, np.isin(vs[0], self.agent_floor) & vs[1] == agent_location[0] & vs[2] == agent_location[1]])
        self.goal_spawn_locations = self._to_flat(vs[:, np.isin(vs[0], self.goal_floor)]) if goal_location is None else self._to_flat(vs[:, np.isin(vs[0], self.goal_floor) & vs[1] == goal_location[0] & vs[2] == goal_location[1]])
        # Internal agent and goal state
        self.agent = np.zeros(self.num_envs, dtype=int)
        self.goal = np.zeros(self.num_envs, dtype=int)
        # Seed if provided
        self.seed(seed)

    def set_obs_fn(self, obs_n: int):
        if obs_n == 0: self._obs = self._discrete_obs
        elif obs_n == 1: self._obs = self._hansen_obs
        else:
            assert (obs_n % 2) == 1
            self._obs = self._grid_obs

    def seed(self, seed: Optional[int] = None):
        """Set internal seed (returns sampled seed if none provided)"""
        self.rng, seed = np_random(seed)
        return seed

    def reset(self):
        """Reset all environments, return observation"""
        self._masked_reset(np.ones(self.num_envs, dtype=bool))
        return self._obs()

    def _masked_reset(self, mask: np.ndarray):
        """Reset some environments, no return"""
        b = mask.sum()
        if b:
            self.elapsed[mask] = 0
            self.goal[mask] = self.rng.choice(self.goal_spawn_locations, size=b)
            self.agent[mask] = self.rng.choice(self.agent_spawn_locations, size=b)
            # Resample agent coordinates until we don't start on a goal
            while (m := mask & (self.agent == self.goal)).any():
                self.agent[m] = self.rng.choice(self.agent_spawn_locations, size=m.sum())

    def step(self, actions: Union[np.ndarray, Sequence[int]]):
        """Step in the environment"""
        self.elapsed += 1
        # Handle movement
        a = vectorized_multinomial_with_rng(self.action_probability_matrix[np.array(actions)], self.rng)  # Sample action failures
        flat_a = self.flat_actions[a]  # Convert to flat coordinates
        new_loc = self.agent + flat_a; moved = self._check_bounds(new_loc)
        self.agent[moved] = new_loc[moved]  # Move agent
        us, ds = self.flat_grid[self.agent] == GR_CNST.stair_up, self.flat_grid[self.agent] == GR_CNST.stair_down
        self.agent[moved & us] += self.flat_actions[-2]
        self.agent[moved & ds] += self.flat_actions[-1]
        # Handle reward
        r = np.zeros(self.num_envs, dtype=np.float32)
        d = (self.agent == self.goal)
        r[d] += 1.
        r[~moved] += self.wall_reward
        d |= self.elapsed > self.time_limit
        self._masked_reset(d)
        return self._obs(), r, d, [{}] * self.num_envs


    def _check_bounds(self, proposed_locations: np.ndarray) -> np.ndarray:
        valid = (0 <= proposed_locations) & (proposed_locations < self.flat_grid.size)  # In bounds
        valid[valid] = self.flat_grid[proposed_locations[valid]] != GR_CNST.wall
        return valid

    def _obs(self) -> np.ndarray:
        """"""
        ...

    def _discrete_obs(self) -> np.ndarray:
        """Discrete observation via lookup table"""
        return self.f_coord_to_e_coord[self.agent]

    def _hansen_obs(self) -> np.ndarray:
        """Hansen-style observations"""
        hgrid = self.flat_grid[(self.agent[:, None] + self.flat_actions[None, :-2]).reshape(-1)].reshape(-1, 4)  # Get adjacent squares for each agent
        # Re-index some stuff (alias stairs, then remove "agent" observation
        hgrid[hgrid == GR_CNST.stair_down] = GR_CNST.stair_up
        hgrid[hgrid > GR_CNST.empty] -= 1
        return hgrid.dot(HANSEN_MULTIPLIERS)

    def _grid_obs(self, obs_n: int = 3) -> np.ndarray:
        """Grid observations around the agent. Need to make sure all on same floor"""
        ag = self._to_grid(self.agent)
        offset = obs_n % 2
        g = np.mgrid[:1, ]
        return np.array([0])


    def render(self, mode="human"):
        """Render environment as an rgb array, with highlighting of agent's view. If "human", render with pygame"""
        ...