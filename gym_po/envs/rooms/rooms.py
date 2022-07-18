from functools import partial
from typing import Tuple, Optional, Union, Sequence, Callable
import numpy as np
import gym
from gym.core import ActType, ObsType
from gym.utils import seeding
from gym.vector.utils import batch_space

from .layouts import *
from .utils import *
from .actions import *
from .observations import *


def get_observation_space_and_function(obs_type: str, grid: np.ndarray, obs_n: int) -> Tuple[gym.Space, Callable[[np.ndarray, np.ndarray], np.ndarray]]:
    """Return space and an observation function"""
    is_vector = 'vector' in obs_type
    has_goal = 'goal' in obs_type
    a_max = np.array(grid.shape) - 2  # Max value of agent's observation
    if 'room' in obs_type:  # No continuous variant
        n = get_number_abstract_states(grid)
        if has_goal:  # Discrete state for agent AND goal combined
            space = gym.spaces.Discrete(int(n ** 2))
            obs = lambda ayx, gyx: grid[tuple(ayx.T)] + n * grid[tuple(gyx.T)]
        else:
            space = gym.spaces.Discrete(int(n))
            obs = lambda ayx, gyx: grid[tuple(ayx.T)]
    elif 'mdp' in obs_type:
        if is_vector:  # Vector observation of position(s)
            if has_goal:
                space = gym.spaces.Box(1, np.tile(a_max, 2), (4,), dtype=int)
                obs = lambda ayx, gyx: np.concatenate((ayx, gyx), -1)
            else:
                space = gym.spaces.Box(1, a_max, (2,), dtype=int)
                obs = lambda ayx, gyx: ayx
        else:
            n, state_grid = get_number_discrete_states_and_conversion(grid)
            if has_goal:
                space = gym.spaces.Discrete(int(n ** 2))
                obs = lambda ayx, gyx: state_grid[tuple(ayx.T)] + n * state_grid[tuple(gyx.T)]
            else:
                space = gym.spaces.Discrete(int(n))
                obs = lambda ayx, gyx: state_grid[tuple(ayx.T)]
    elif 'hansen' in obs_type:  # No continuous. Cont converts to outputting a vector instead of scalar
        base_n = 8 if '8' in obs_type else 4
        if is_vector:
            if has_goal:
                space = gym.spaces.Box(0, 2, (base_n,), dtype=int)
                obs = lambda ayx, gyx: get_hansen_vector_obs(ayx, grid, gyx, base_n)
            else:
                space = gym.spaces.Box(0, 1, (base_n,), dtype=int)
                obs = lambda ayx, gyx: get_hansen_vector_obs(ayx, grid, None, base_n)
        else: # No goal
            space = gym.spaces.Discrete(int(2 ** base_n * (base_n + 1)))
            obs = lambda ayx, gyx: get_hansen_obs(ayx, grid, gyx, base_n)
    elif 'grid' in obs_type: # No continuous, no has goal
        space = gym.spaces.Box(0, 2, (obs_n, obs_n), dtype=int)
        obs = lambda ayx, gyx: get_grid_obs(ayx, grid, gyx, obs_n)
    else:
        raise NotImplementedError('Observation type not recognized')
    return space, obs


class Rooms(gym.Env):
    """Basic ROOMS domain adapted from "Markovian State and Action Abstraction"
    
    See https://github.com/aijunbai/hplanning for official repo
    This is a vectorized version of the ROOMs domain.
    """
    metadata = {"name": "Rooms", "render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    def __init__(self, num_envs: int, layout: str = '4', time_limit: int = 500,
                 obs_type: str = 'mdp', obs_n: int = 3, action_failure_probability: float = 0.2, action_type: str = 'ordinal',
                 agent_xy: Optional[Sequence[int]] = None, goal_xy: Optional[Sequence[int]] = (0, 0),
                 step_reward: float = 0., wall_reward: float = 0., goal_reward: float = 1.,
                 **kwargs):
        """
        Args:
            num_envs: Number of environments
            layout: Key to layouts, one of '1', '2', '4', '4b', '8', '8b', '10', '10b', '16', '16b', '32', '32b'
            time_limit: Max time before episode terminates
            obs_type: Type of observation. One of 'discrete', 'hansen', 'hansen8', 'vector_hansen', 'vector_hansen8', 'room', 'grid'
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
        self.metadata['name'] += f'__{layout}__{action_type}__{obs_type}'
        grid = np_to_grid(layout_to_np(LAYOUTS[layout]))
        if 'b' in layout: layout = layout[:-1]  # Remove b for later indexing
        self.grid = grid
        self.gridshape = np.array(grid.shape)
        self.single_observation_space, self._get_obs = get_observation_space_and_function(obs_type, self.grid, obs_n)
        self.valid_states = np.flatnonzero(grid >= 0)  # Places where we can put goal or agent
        self.rng, _ = seeding.np_random()

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

    def seed(self, seed: Optional[int] = None):
        """Set internal seed (returns sampled seed if none provided)"""
        self.rng, seed = seeding.np_random(seed)
        return seed

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[ObsType, tuple[ObsType, dict]]:
        """Reset all environments, set seed if given"""
        if seed is not None: self.seed(seed)
        self.elapsed = np.zeros(self.num_envs, int)
        self.goal_yx = self._sample_goal(self.num_envs, self.rng)
        self.agent_yx = self._sample_agent(self.num_envs, self.rng)
        obs = self._get_obs(self.agent_yx, self.goal_yx)
        return obs

    def _reset_some(self, mask: np.ndarray):
        """Reset only a subset of environments"""
        if b := mask.sum():
            self.elapsed[mask] = 0
            self.goal_yx[mask] = self._sample_goal(b, self.rng)
            self.agent_yx[mask] = self._sample_agent(b, self.rng)

    def step(self, action: ActType) -> Tuple[ObsType, np.ndarray, np.ndarray, Union[dict, list]]:
        """Step in environment

        Sample random action failure. Move agent(s) where move is valid.
        Check if we reached goal. Update with step, wall, and goal rewards.
        """
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
        return self._get_obs(self.agent_yx, self.goal_yx), r, d, [{}] * self.num_envs

    def _out_of_bounds(self, proposed_yx: np.ndarray):
        """Return whether given coordinates correspond to empty/goal square.

        Rooms are surrounded by walls, so only need to check this"""
        # oob = (proposed_yx >= self.gridshape[None, :]).any(-1) | (proposed_yx < 0).any(-1)
        # oob[~oob] = self.grid[tuple(proposed_yx[~oob].T)] == -1
        # return oob
        return self.grid[tuple(proposed_yx.T)] == -1
