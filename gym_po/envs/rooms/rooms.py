from functools import partial
from typing import Tuple, Optional, Union, Sequence
import numpy as np
import gym
from gym.core import ActType, ObsType
from gym.utils import seeding
from gym.vector.utils import batch_space

from .layouts import *
from .utils import *
from .actions import *
from .observations import *


class Rooms(gym.Env):
    """Basic ROOMS domain adapted from "Markovian State and Action Abstraction"
    
    See https://github.com/aijunbai/hplanning for official repo
    This is a vectorized version of the ROOMs domain.
    """
    metadata = {"name": "ROOMS", "render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    def __init__(self, num_envs: int, layout: str = '4', time_limit: int = 500,
                 obs_type: str = 'hansen', obs_n: int = 3, action_failure_probability: float = 1./3, action_type: str = 'ordinal',
                 agent_xy: Optional[Sequence[int]] = None, goal_xy: Optional[Sequence[int]] = (0, 0),
                 step_reward: float = 0., wall_reward: float = 0., goal_reward: float = 1.,
                 ):
        """
        Args:
            num_envs: Number of environments
            layout: Key to layouts, one of '1', '2', '4', '8', '10', '16', '32'
            time_limit: Max time before episode terminates
            obs_type: Type of observation. One of 'discrete', 'hansen', 'hansen8', 'room', 'grid'
                hansen is 4 adjacent <empty|wall|goal>, hansen8 is 8. room treats each room as an obs
            obs_n: Only applies if 'grid' observation. Dictates nxn observation grid (centered on agent)
            action_failure_probability: Likelihood that taking one action fails and chooses another
            action_type: 'ordinal' (8D compass) or 'cardinal (4D compass)
            agent_xy: Optionally, provide a fixed (x, y) agent location used every reset. Defaults to random
            goal_xy: Optionally, provide a fixed (x, y) goal location used every reset.
                If you give invalid coordinate (e.g., (0,0) is a wall), uses default goal from layouts.
            step_reward: Reward for each step
            wall_reward: Reward for hitting a wall
            goal_reward: Reward for reaching goal
        """
        assert layout in LAYOUTS
        grid = np_to_grid(layout_to_np(LAYOUTS[layout]))
        self.grid = grid
        self.gridshape = np.array(grid.shape)
        if obs_type == 'discrete':
            n, state_grid = get_number_discrete_states_and_conversion(grid)
            self.single_observation_space = gym.spaces.Discrete(n)
            self._get_obs = lambda agent_yx, gr, goal: state_grid[agent_yx]
        elif obs_type == 'hansen' or obs_type == 'hansen8':
            base_n = 4 if obs_type == 'hansen' else 8
            hobs_fn = partial(get_hansen_obs, hansen_n=base_n)
            n = base_n ** 3  # base_n squares, 3 possibilities each
            self.single_observation_space = gym.spaces.Discrete(n)
            self._get_obs = lambda agent_yx, gr, goal: hobs_fn(agent_yx, gr, goal)
        elif obs_type == 'room':
            n = get_number_abstract_states(grid)
            self.single_observation_space = gym.spaces.Discrete(n)
            self._get_obs = lambda agent_yx, gr, goal: grid[agent_yx]
        else:  # Grid observations
            self.single_observation_space = gym.spaces.Box(0, 2, (obs_n, obs_n), dtype=int)
            gobs_fn = partial(get_grid_obs, n=obs_n)
            self._get_obs = lambda agent_yx, gr, goal: gobs_fn
        self.valid_states = np.flatnonzero(grid >= 0)  # Places where we can put goal or agent

        self.actions = ACTIONS_CARDINAL if action_type == 'cardinal' else ACTIONS_ORDINAL
        # Boilerplate for vector environment
        self.num_envs = num_envs
        self.is_vector_env = True
        self.single_action_space = gym.spaces.Discrete(self.actions.shape[0])
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.observation_space = batch_space(self.single_observation_space, num_envs)

        # Constants
        self.time_limit = time_limit
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.wall_reward = wall_reward

        # Random or fixed goal/agent
        if goal_xy is not None:
            goal_yx = tuple(reversed(goal_xy))  # (x,y) to (y,x)
            if grid[goal_yx] < 0: goal_yx = tuple(reversed(ENDS[layout]))
            goal_yx = np.array(goal_yx)
            self._sample_goal = lambda b, rng: np.full((b, 2), goal_yx, dtype=int)
        else: self._sample_goal = lambda b, rng: np.array(np.unravel_index(rng.choice(self.valid_states, b), self.grid.shape)).swapaxes(0,1)
        if agent_xy is not None:
            agent_yx = tuple(reversed(agent_xy))
            agent_yx = np.array(agent_yx)
            if grid[agent_yx] < 0: agent_yx = tuple(reversed(STARTS[layout]))
            self._sample_agent = lambda b, rng: np.full((b, 2), agent_yx, dtype=int)
        else: self._sample_agent = lambda b, rng: np.array(np.unravel_index(rng.choice(self.valid_states, b), self.grid.shape)).swapaxes(0,1)
        self.action_matrix = create_action_probability_matrix(self.actions.shape[0], action_failure_probability)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[ObsType, tuple[ObsType, dict]]:
        self.rng, seed = seeding.np_random(seed)
        self.elapsed = np.zeros(self.num_envs, int)
        self.goal_yx = self._sample_goal(self.num_envs, self.rng)
        self.agent_yx = self._sample_agent(self.num_envs, self.rng)
        obs = self._get_obs(self.agent_yx, self.grid, self.goal_yx)
        return obs

    def _reset_some(self, mask: np.ndarray):
        """Reset only a subset of environments"""
        self.elapsed[mask] = 0
        self.goal_yx[mask] = self._sample_goal(self.num_envs, self.rng)
        self.agent_yx[mask] = self._sample_agent(self.num_envs, self.rng)

    def step(self, action: ActType) -> Tuple[ObsType, np.ndarray, np.ndarray, dict]:
        self.elapsed += 1
        # Movement
        a = vectorized_multinomial_with_rng(self.action_matrix[action], self.rng)
        proposed_yx = self.agent_yx + self.actions[a]
        oob = self._out_of_bounds(proposed_yx)
        self.agent_yx[~oob] = proposed_yx[~oob]
        # Reward
        r = np.zeros(self.num_envs, dtype=np.float32)
        d = (self.agent_yx == self.goal_yx).all(-1)
        r += self.step_reward
        r[oob] = self.wall_reward
        r[d] = self.goal_reward
        d |= self.elapsed > self.time_limit
        self._reset_some(d)
        return self._get_obs(self.agent_yx, self.grid, self.goal_yx), r, d, {}

    def _out_of_bounds(self, proposed_yx: np.ndarray):
        oob = (proposed_yx >= self.gridshape[None, :]).any(-1) | (proposed_yx < 0).any(-1)
        oob[~oob] = self.grid[proposed_yx[~oob]] != -1
        return oob




if __name__ == "__main__":
    map = '32'
    test1 = layout_to_np(LAYOUTS[map])
    test2 = np_to_grid(test1)
    agent_yx = np.random.randint(1, tuple(t - 1 for t in test2.shape), size=(16, 2))
    goal_yx = np.random.randint(1, tuple(t - 1 for t in test2.shape), size=(16, 2))
    coords = (agent_yx[:, None] + ACTIONS_ORDINAL[None, :])
    is_goal = (coords == goal_yx[:, None]).all(-1)
    squares = test2[tuple(coords.transpose(2,0,1))]
    squares += 1
    squares[squares > 0] = 1
    squares[is_goal] = 2
    MULTIPLIERS = np.array([3 ** i for i in range(ACTIONS_ORDINAL.shape[0])])  # 3 possibilities for adjacent observations
    print(3)
