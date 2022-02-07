from typing import Sequence, Tuple, Callable, Optional
import numpy as np
import gym
from gym.utils.seeding import np_random

from .grid_utils import WALLS, DIRECTIONS_2D_NP
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


# Constant colors for each object
class GR_CNST:
    empty = COLORS.white
    wall = COLORS.black
    agent = COLORS.green
    goal = COLORS.blue
    stair_up = COLORS.gray_light
    stair_down = COLORS.gray_dark


def convert_str_map_to_walled_np_str(map: Sequence[str]) -> np.ndarray:
    """Return the full map convert (for image and navigation)"""
    return np.pad(np.asarray(map, dtype='c').astype(str), 1, constant_values='|')


def generate_layout(map: Sequence[str], grid_z: int = 1) -> np.ndarray:
    """Return full zyx(rgb) map layout

    Args:
        map: Raw string map
        grid_z: Number of floors
    Returns:
        img_map (z,y,x,3) unscaled img
    """
    bordered_map = convert_str_map_to_walled_np_str(map)  # Get bordered version
    img_map = np.zeros((grid_z, *bordered_map.shape, 3), dtype=np.uint8)  # Get an unscaled image version for each floor, walls are already filled in
    img_map[:, bordered_map == ' '] = GR_CNST.empty  # Fill in floors
    if grid_z == 1: return img_map  # No stairs for one floor
    img_map[0, bordered_map == 'U'] = GR_CNST.stair_up
    img_map[1, bordered_map == 'D'] = GR_CNST.stair_down
    if grid_z == 2: return img_map  # One stair per floor
    img_map[1:-1, bordered_map == 'U'] = GR_CNST.stair_up
    img_map[1:-1, bordered_map == 'D'] = GR_CNST.stair_up
    return img_map





def compute_obs_space(img_map: np.ndarray, hansen: bool = False) -> int:
    """Compute discrete observation space

    Args:
        layout: Full map (floors, y, x, 3), can be bordered or now
        hansen: Use hansen (adjacent empty/wall/stair/goal only) observations
    Returns:
        n: Number of possible discrete observations
    """
    return int(4 ** 4) if hansen else img_map[...,].nonzero().sum().item()


class MultistoryFourRoomsVecEnv(gym.Env):
    """Vectorized Multistory fourrooms environment"""
    metadata = {"name": "MultistoryFourRooms", "render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    ACTIONS = DIRECTIONS_2D_NP[:, :4]
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
            agent_floor: Floor where agent can spawn (defaults to bottom)
            goal_floor: Floor where goal can spawn (defaults to top)
            agent_location: If provided, a fixed location where agent spawns each episode (defaults random)
            goal_location: If provided, a fixed location where goal spawns each episode (defaults to east hallway)
            wall_reward: Reward for hitting a wall, should be negative (defaults to 0
        """
        # VectorEnv
        self.num_envs = num_envs
        self.is_vector_env = True
        # Time limit
        self.time_limit = time_limit
        self.elapsed = np.zeros(num_envs, dtype=int)
        # Agent and goal sampling
        self.agent_floor = agent_floor
        self.goal_floor = goal_floor
        self.fixed_agent_location = agent_location
        self.fixed_goal_location = goal_location
        # Observation space
        self.discrete = obs_n == 0
        self.hansen = obs_n == 1
        self.img_grid = generate_layout(map, grid_z)
        if (obs_n < 3): o_n = compute_obs_space(self.img_grid)
        self.single_observation_space


    def seed(self, seed: Optional[int] = None):
        self.rng, seed = np_random(seed)
        return seed

if __name__ == "__main__":
    e = MultistoryFourRoomsVecEnv(8, 3)