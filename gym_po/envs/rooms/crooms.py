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


def get_room_obs(agent_yx: np.ndarray, room_or_state_grid: np.ndarray, goal_yx: np.ndarray, cell_size: float = 1.) -> np.ndarray:
    """Get room/state that agent(s) is in"""
    return room_or_state_grid[tuple(coord_to_grid(agent_yx, cell_size).T)]


def get_lidar_obs(agent_yx: np.ndarray, grid: np.ndarray, goal_yx: np.ndarray, n_bins: int = 8, obs_m: float = 3., cell_size: float = 1.) -> np.ndarray:
    """Get rangefinder observation from agent"""
    angles = np.arange(0, n_bins, 360 / n_bins)
    # Fill in goal
    relative_yx = goal_yx - agent_yx
    goal_in_range = ((goal_yx - agent_yx) ** 2).sum(-1) <= obs_m  # Goal is within observation range
    obs = np.zeros((agent_yx.shape[0], n_bins + 2))
    obs[goal_in_range] = relative_yx
    # TODO: Fill in distance to nearest wall in each direction
    return obs

class CRooms(gym.Env):
    """Basic CROOMS domain adapted from "Markovian State and Action Abstraction"
    
    See https://github.com/aijunbai/hplanning for official repo
    This is a vectorized version of the C-ROOMs domain.
    """
    metadata = {"name": "C-Rooms", "render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    def __init__(self, num_envs: int, layout: str = '4', time_limit: int = 500, use_velocity: bool = False, cell_size: float = 1.,
                 obs_type: str = 'lidar', obs_m: int = 3, obs_bins: int = 8,
                 action_failure_probability: float = 0.2, action_type: str = 'ordinal', action_std: float = 0.2,
                 agent_xy: Optional[Sequence[float]] = None, goal_xy: Optional[Sequence[float]] = (0., 0.),
                 step_reward: float = 0., wall_reward: float = 0., goal_reward: float = 1.,
                 ):
        """
        Args:
            num_envs: Number of environments
            layout: Key to layouts, one of '1', '2', '4', '8', '10', '16', '32'
            time_limit: Max time before episode terminates
            use_velocity: If true, agent actions alter velocity which adjusts position. If false, actions just move agent without inertia.
            cell_size: Size of a grid cell, in meters
            obs_type: Type of observation.
                'discrete': Integer grid square
                'room': Integer room number
                'lidar': [bins+2,] vector of range to nearest wall, then 2D for relative xy position of goal
            obs_m: Range of observation (m). If wall/goal is out of range, that bin will be 0
            obs_bins: Number of observation bins

            action_failure_probability: Likelihood that taking one action fails and chooses another
            action_type: 'ordinal' (8D compass) or 'cardinal' (4D compass) or 'xy' (2D continuous)
            action_std: Standard deviation of action error (sampled from normal distribution)
            agent_xy: Optionally, provide a fixed (x, y) agent location used every reset. Defaults to random
            goal_xy: Optionally, provide a fixed (x, y) goal location used every reset.
                If you give invalid coordinate (e.g., (0,0) is a wall), uses default goal from layouts.
            step_reward: Reward for each step
            wall_reward: Reward for hitting a wall
            goal_reward: Reward for reaching goal
        """
        assert layout in LAYOUTS
        self.metadata['name'] += f'_{layout}_{action_type}'
        grid = np_to_grid(layout_to_np(LAYOUTS[layout]))
        self.grid = grid
        self.gridshape = np.array(grid.shape)
        if obs_type == 'discrete':
            n, state_grid = get_number_discrete_states_and_conversion(grid)
            self.single_observation_space = gym.spaces.Discrete(n)
            self._get_obs = lambda agent_yx, gr, goal: partial(get_room_obs, cell_size=cell_size)(agent_yx, state_grid, goal)
        elif obs_type == 'room':
            n = get_number_abstract_states(grid)
            self.single_observation_space = gym.spaces.Discrete(n)
            self._get_obs = lambda agent_yx, gr, goal: partial(get_room_obs, cell_size=cell_size)(agent_yx, gr, goal)
        else:
            self.single_observation_space = gym.spaces.Box(0, obs_m, (obs_bins + 2,))
            self._get_obs = lambda agent_yx, gr, goal: ...
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
        obs = self._get_obs(self.agent_yx, self.grid, self.goal_yx)
        return obs

    def _reset_some(self, mask: np.ndarray):
        """Reset only a subset of environments"""
        if b := mask.sum():
            self.elapsed[mask] = 0
            self.goal_yx[mask] = self._sample_goal(b, self.rng)
            self.agent_yx[mask] = self._sample_agent(b, self.rng)

    def step(self, action: ActType) -> Tuple[ObsType, np.ndarray, np.ndarray, dict]:
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
        return self._get_obs(self.agent_yx, self.grid, self.goal_yx), r, d, {}

    def _out_of_bounds(self, proposed_yx: np.ndarray):
        """Return whether given coordinates correspond to empty/goal square.

        Rooms are surrounded by walls, so only need to check this"""
        # oob = (proposed_yx >= self.gridshape[None, :]).any(-1) | (proposed_yx < 0).any(-1)
        # oob[~oob] = self.grid[tuple(proposed_yx[~oob].T)] == -1
        # return oob
        return self.grid[tuple(proposed_yx.T)] == -1