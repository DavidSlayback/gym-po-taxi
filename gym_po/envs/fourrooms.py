import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
from gym.vector.utils import batch_space
from gym.utils.seeding import np_random

# Up, right, down, left
ACTIONS = np.array([
    [-1, 0],
    [0, 1],
    [1, 0],
    [0, -1]
], dtype=int)

# Basic 13x13 FourRooms grid
MAP = np.array([
    [0]*13,
    [0]+([1]*5) + [0] + ([1]*5) + [0],
    [0]+([1]*5) + [0] + ([1]*5) + [0],
    [0]+([1]*11) + [0],
    [0]+([1]*5) + [0] + ([1]*5) + [0],
    [0]+([1]*5) + [0] + ([1]*5) + [0],
    [0, 0, 1] + ([0]*4) + ([1]*5) + [0],
    [0] + ([1]*5) + ([0]*3) + [1] + ([0]*3),
    [0]+([1]*5) + [0] + ([1]*5) + [0],
    [0]+([1]*5) + [0] + ([1]*5) + [0],
    [0]+([1]*11) + [0],
    [0]+([1]*5) + [0] + ([1]*5) + [0],
    [0]*13,
], dtype=int)

y, x = MAP.shape
empty_locs = MAP.nonzero()
flat_locs = np.ravel_multi_index(empty_locs, (y,x))

def base_encode(r, c):
    """Encode row, col as discrete state"""


class FourRoomsVecEnv(Env):
    def __init__(self, num_envs: int, time_limit: int = 10000, seed=None):
        # Preliminaries
        self.num_envs = num_envs
        self.is_vector_env = True
        self.time_limit = time_limit

        # Actions
        self.single_action_space = Discrete(4)
        self.action_space = batch_space(self.single_action_space, num_envs)

        # Observations
        self.single_observation_space = Discrete(empty_locs.sum())
        self.observation_space = batch_space(self.single_observation_space, num_envs)

        self.seed(seed)
        # States
        self.s = np.zeros(self.num_envs, int)
        self.elapsed = np.zeros(self.num_envs, int)

    def seed(self, seed=None):
        self.rng, seed = np_random(seed)
        return seed

    def reset(self):
        self._reset_mask(np.ones(self.num_envs, bool))
        return self._obs()

    def _reset_mask(self, mask: np.ndarray):
        b = mask.sum()
        if b:
            self.s[mask] = 0
            self.elapsed[mask] = 0

    def _obs(self):
        return self.s

    def step(self, actions: np.ndarray):
        ...

if __name__ == "__main__":
    print(3)
    print(4)
