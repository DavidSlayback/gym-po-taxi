

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

# Up, right, down, left, upstairs, downstairs (z, y, x)
ACTIONS = np.array([
    [0, -1, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 0, -1],
    [1, 0, 0],
    [-1, 0, 0]
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
y, x = MAP.shape
NE = (1, 11)  # NE stairs are upstairs
SW = (11, 1)  # SW stairs are downstairs

# Add stairs to bottom, mid, and top floors
B_MAP, M_MAP, T_MAP = [MAP.copy()] * 3
B_MAP[NE] = STAIR; M_MAP[NE] = M_MAP[SW] = STAIR; T_MAP[SW] = STAIR

def generate_layout(grid_z: int = 1):
    """Generate multi-story layout (z, y, x)"""
    if grid_z == 1:
        return MAP.copy()[None, ...]
    elif grid_z == 2:
        return np.stack((B_MAP, T_MAP), axis=0)
    else:
        return np.stack((B_MAP,*(M_MAP for _ in range(grid_z-2)), T_MAP), axis=0)


def generate_flat_actions(grid_z: int = 1):
    """Generate flattened versions of actions"""
    return np.ravel_multi_index((ACTIONS+1).T, (grid_z, y, x)) - np.ravel_multi_index((1, 1, 1), (grid_z, y, x))


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


class MultistoryFourRoomsVecEnv(Env):
    """Discrete multistory fourrooms environment"""
    def __init__(self, num_envs: int, grid_z: int = 1, time_limit: int = 10000, seed=None,
                 fixed_goal: int = 0, action_failure_probability: float = (1./3),
                 wall_reward: float = 0.):
        # Preliminaries
        self.num_envs = num_envs
        self.is_vector_env = True
        self.time_limit = time_limit
        self.wall_reward = wall_reward

        # Actions
        self.single_action_space = Discrete(4)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.aprob_with_failure = action_probability_matrix(action_failure_probability, 4)
        # Grid
        self.grid = generate_layout(grid_z)
        self.EMPTY_LOCS = (self.grid == EMPTY).nonzero()
        self.grid_flat = self.grid.ravel()
        self.FLAT_EMPTY = np.ravel_multi_index(EMPTY_LOCS, (grid_z, y, x))
        self.actions = generate_flat_actions(grid_z)

        # Observations
        self._ns = len(self.FLAT_EMPTY)
        self.single_observation_space = Discrete(self._ns)
        self.observation_space = batch_space(self.single_observation_space, num_envs)

        self.seed(seed)
        # States
        self.agent = np.zeros(self.num_envs, int)
        self.elapsed = np.zeros(self.num_envs, int)
        self.goal_fn = lambda b: np.full(b, fixed_goal, dtype=int) if fixed_goal else lambda b: self.rng.choice(self.FLAT_EMPTY, b)
        self.goal = np.full(self.num_envs, fixed_goal, dtype=int)

    def seed(self, seed=None):
        self.rng, seed = np_random(seed)
        return seed

    def set_goal(self, goal):
        self.goal_fn = lambda b: np.full(b, goal, dtype=int)
        self.goal[:] = self.goal_fn(self.num_envs)

    def reset(self):
        self._reset_mask(np.ones(self.num_envs, bool))
        return self._obs()

    def _reset_mask(self, mask: np.ndarray):
        b = mask.sum()
        if b:
            self.goal[mask] = self.goal_fn(b)
            # Make sure agent doesn't start on goal
            self.agent[mask] = self.rng.choice(self.EMPTY_LOCS, b)
            while (m := mask & (self.agent == self.goal)).any():
                self.agent[m] = self.rng.choice(self.FLAT_EMPTY, m.sum())
            self.elapsed[mask] = 0

    def _obs(self):
        return self.agent

    def step(self, actions: np.ndarray):
        self.elapsed += 1
        rnd = self.rng.random(self.num_envs)
        a = vectorized_multinomial(self.aprob_with_failure[np.array(actions)], rnd)
        new_loc = self.agent + self.actions[a]
        b = self._check_bounds(new_loc)
        self.agent[b] = new_loc[b]  # Update where move was valid
        r = np.zeros(self.num_envs, dtype=np.float32)
        d = (self.agent == self.goal)  # Done where we reached goal
        r[d] = 1.
        r[~b] = self.wall_reward  # Potentially penalize hitting walls
        d |= self.elapsed > self.time_limit  # Also done where we're out of time
        self._reset_mask(d)
        return self._obs, r, d, {}

    def _check_bounds(self, proposed_locations: np.ndarray):
        valid = (0 <= proposed_locations) & (proposed_locations < self.grid_flat.size)  # In bounds
        valid[valid] = self.grid_flat[proposed_locations[valid]] != WALL
        return valid

if __name__ == "__main__":
    e = MultistoryFourRoomsVecEnv(8, 3)
    print(3)
