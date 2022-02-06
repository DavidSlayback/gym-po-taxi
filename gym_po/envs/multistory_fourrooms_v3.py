from typing import Sequence, Tuple, Callable, Optional
import numpy as np
import gym

from .grid_utils import WALLS, DIRECTIONS_2D_NP

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


def convert_str_map_to_walled_np_str(map: Sequence[str]) -> np.ndarray:
    """Return the full map convert (for image and navigation)"""
    return np.pad(np.asarray(map, dtype='c').astype(str), 1, constant_values='|')


def compute_obs_space(layout: np.ndarray, hansen: bool = False) -> int:
    """Compute discrete observation space

    Args:
        layout: Full map (floors, y, x), can be bordered or now
        hansen: Use hansen (adjacent walls only) observations
    Returns:
        n: Number of possible discrete observations
    """
    return int(2 ** 4) if hansen else int((~np.isin(layout, WALLS)).sum().item())


class MultistoryFourRoomsVecEnv(gym.Env):
    """Vectorized Multistory fourrooms environment"""
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    ACTIONS = DIRECTIONS_2D_NP[:, :4]

    def __init__(self, num_envs: int, grid_z: int = 1, map: Sequence[str] = BASE_FOURROOMS_MAP_WITH_STAIRS,
                 seed: Optional[int] = None, time_limit: int = 100,
                 action_failure_probability: float = 1./3, agent_floor: int = 0, goal_floor: int = -1,
                 agent_location: Optional[Tuple[int, int]] = None, goal_location: Optional[Tuple[int, int]] = None,
                 wall_reward: float = 0.):
        """Create a multistory four rooms environment

        Args:
            num_envs: Number of vectorized environments
            grid_z: Number of floors
            map: Base (unbordered) map for floors. 'U' is upstairs, 'D' is downstairs
            seed: Seed to use
            time_limit: Max number of timesteps before an environment resetes
            action_failure_probability: Probability that a chosen action will fail
            agent_floor: Floor where agent can spawn (defaults to bottom)
            goal_floor: Floor where goal can spawn (defaults to top)
            agent_location: If provided, a fixed location where agent spawns each episode (defaults random)
            goal_location: If provided, a fixed location where goal spawns each episode (defaults to east hallway)
            wall_reward: Reward for hitting a wall, should be negative (defaults to 0
        """
        pass

if __name__ == "__main__":
    bmap, tmap, cc = convert_str_map_to_walled_np_str(BASE_FOURROOMS_MAP_WITH_STAIRS)