__all__ = ['FourRoomsVecEnv']

import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
from gym.vector.utils import batch_space
from gym.utils.seeding import np_random
from typing import Optional

# Up, right, down, left
ACTIONS = np.array([
    [-1, 0],
    [0, 1],
    [1, 0],
    [0, -1]
], dtype=int)

# Basic 13x13 FourRooms grid. 0 is wall. 1 is empty
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
FLAT_MAP = MAP.ravel()

y, x = MAP.shape
# Actions in terms of discrete state
FLAT_ACTIONS = np.ravel_multi_index((ACTIONS+1).T, (y,x)) - np.ravel_multi_index((1,1), (y,x))
EMPTY_LOCS = MAP.nonzero()
FLAT_EMPTY_LOCS = np.ravel_multi_index(EMPTY_LOCS, (y, x))
GRID_ENCODER = MAP.copy(); GRID_ENCODER[GRID_ENCODER==0] = -1
GRID_ENCODER[EMPTY_LOCS] = np.arange(len(FLAT_EMPTY_LOCS))

def base_encode(r, c):
    """Encode row, col as discrete state"""
    return GRID_ENCODER[r, c]


def base_decode(s):
    """Get row, col from discrete state"""
    return EMPTY_LOCS[0][s], EMPTY_LOCS[1][s]


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


class FourRoomsVecEnv(Env):
    def __init__(self, num_envs: int, time_limit: int = 10000, seed=None,
                 fixed_goal: int = 62, action_failure_probability: float = (1./3),
                 wall_reward: float = 0.):
        # Preliminaries. 62 is E hallway
        self.num_envs = num_envs
        self.is_vector_env = True
        self.time_limit = time_limit
        self.wall_reward = wall_reward

        # Actions
        self.single_action_space = Discrete(4)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.aprob_with_failure = action_probability_matrix(action_failure_probability, 4)

        # Observations
        self._ns = len(FLAT_EMPTY_LOCS)
        self.single_observation_space = Discrete(self._ns)
        self.observation_space = batch_space(self.single_observation_space, num_envs)

        self.seed(seed)
        # States
        self.agent = np.zeros(self.num_envs, int)
        self.elapsed = np.zeros(self.num_envs, int)
        self.goal_fn = lambda b: np.full(b, fixed_goal, dtype=int) if fixed_goal else lambda b: self.rng.choice(FLAT_EMPTY_LOCS, b)
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
            self.agent[mask] = self.rng.choice(FLAT_EMPTY_LOCS, b)
            while (m := mask & (self.agent == self.goal)).any():
                self.agent[m] = self.rng.choice(FLAT_EMPTY_LOCS, m.sum())
            self.elapsed[mask] = 0

    def _obs(self):
        return self.agent

    def step(self, actions: np.ndarray):
        self.elapsed += 1
        rnd = self.rng.random(self.num_envs)
        a = vectorized_multinomial(self.aprob_with_failure[np.array(actions)], rnd)
        new_loc = self.agent + FLAT_ACTIONS[a]
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
        valid = (0 <= proposed_locations) & (proposed_locations < FLAT_MAP.size)  # In bounds
        valid[valid] = FLAT_MAP[proposed_locations[valid]]
        return valid

if __name__ == "__main__":
    e = FourRoomsVecEnv(8)
    o = e.reset()
    for t in range(100000):
        o, r, d, info = e.step(e.action_space.sample())
        if d.any(): print(r[d])
    print(4)