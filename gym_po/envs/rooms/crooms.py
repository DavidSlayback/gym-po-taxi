from typing import Tuple, Optional, Union, Sequence, Callable

import gymnasium
import numpy as np
from numpy.typing import NDArray
from gymnasium.core import ActType, ObsType
from gymnasium.utils import seeding
from gymnasium.vector.utils import batch_space

from .action_utils import *
from .layouts import *
from .observations import *
from .utils import *


def get_observation_space_and_function(
    obs_type: str, grid: NDArray[int], obs_m: int, cell_size: float = 1.0
) -> Tuple[gymnasium.Space, Callable[[NDArray, NDArray], NDArray]]:
    """Return space and an observation function"""
    is_vector = "vector" in obs_type
    has_goal = "goal" in obs_type
    a_max = np.array(grid.shape) - 1 - 1e-6  # Max value of agent's observation
    if "room" in obs_type:  # No continuous variant
        n = get_number_abstract_states(grid)
        if has_goal:  # Discrete state for agent AND goal combined
            space = gymnasium.spaces.Discrete(int(n**2))
            obs = (
                lambda ayx, gyx: grid[tuple(coord_to_grid(ayx, cell_size).T)]
                + n * grid[tuple(coord_to_grid(gyx, cell_size).T)]
            )
        else:
            space = gymnasium.spaces.Discrete(int(n))
            obs = lambda ayx, gyx: grid[tuple(coord_to_grid(ayx, cell_size).T)]
    elif "mdp" in obs_type:
        if is_vector:  # Vector observation of position(s)
            if has_goal:
                space = gymnasium.spaces.Box(1.0, np.tile(a_max, 2), (4,))
                obs = lambda ayx, gyx: np.concatenate((ayx, gyx), -1)
            else:
                space = gymnasium.spaces.Box(1.0, a_max, (2,))
                obs = lambda ayx, gyx: ayx
        else:
            n, state_grid = get_number_discrete_states_and_conversion(grid)
            if has_goal:
                space = gymnasium.spaces.Discrete(int(n**2))
                obs = (
                    lambda ayx, gyx: state_grid[tuple(coord_to_grid(ayx, cell_size).T)]
                    + n * state_grid[tuple(coord_to_grid(gyx, cell_size).T)]
                )
            else:
                space = gymnasium.spaces.Discrete(int(n))
                obs = lambda ayx, gyx: state_grid[
                    tuple(coord_to_grid(ayx, cell_size).T)
                ]
    elif (
        "hansen" in obs_type
    ):  # No continuous. Cont converts to outputting a vector instead of scalar
        base_n = 8 if "8" in obs_type else 4
        if is_vector:
            if has_goal:
                space = gymnasium.spaces.Box(0, 2, (base_n,), dtype=int)
                obs = lambda ayx, gyx: get_hansen_vector_obs(
                    coord_to_grid(ayx, cell_size),
                    grid,
                    coord_to_grid(gyx, cell_size),
                    base_n,
                )
            else:
                space = gymnasium.spaces.Box(0, 1, (base_n,), dtype=int)
                obs = lambda ayx, gyx: get_hansen_vector_obs(
                    coord_to_grid(ayx, cell_size), grid, None, base_n
                )
        else:  # No goal
            space = gymnasium.spaces.Discrete(int(2**base_n * (base_n + 1)))
            obs = lambda ayx, gyx: get_hansen_obs(
                coord_to_grid(ayx, cell_size),
                grid,
                coord_to_grid(gyx, cell_size),
                base_n,
            )
    elif "grid" in obs_type:  # No continuous, no has goal
        space = gymnasium.spaces.Box(0, 2, (obs_m, obs_m), dtype=int)
        obs = lambda ayx, gyx: get_grid_obs(
            coord_to_grid(ayx, cell_size), grid, coord_to_grid(gyx, cell_size), obs_m
        )
    else:
        raise NotImplementedError("Observation type not recognized")
    return space, obs


class CRoomsEnv(gymnasium.Env):
    """Basic CROOMS domain adapted from "Markovian State and Action Abstraction"

    See https://github.com/aijunbai/hplanning for official repo
    This is a vectorized version of the C-ROOMs domain.
    """

    metadata = {
        "name": "CRooms",
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": 10,
    }

    def __init__(
        self,
        num_envs: int,
        layout: str = "4",
        time_limit: int = 500,
        use_velocity: bool = False,
        cell_size: float = 1.0,
        obs_type: str = "mdp",
        obs_m: int = 3,
        action_failure_probability: float = 0.2,
        action_type: str = "yx",
        action_std: float = 0.2,
        action_power: float = 1.0,
        agent_xy: Optional[Sequence[int]] = None,
        goal_xy: Optional[Sequence[int]] = (0, 0),
        step_reward: float = 0.0,
        wall_reward: float = 0.0,
        goal_reward: float = 1.0,
        goal_threshold: float = 0.5,
        render_mode: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            num_envs: Number of environments
            layout: Key to layouts, one of '1', '2', '4', '4b', '8', '8b', '10', '10b', '16', '16b', '32', '32b'
            time_limit: Max time before episode terminates
            use_velocity: If true, agent actions alter velocity which adjusts position. If false, actions just move agent without inertia. TODO: Velocity as part of observation
            cell_size: Size of a grid cell, in meters
            obs_type: Type of observation.
                'discrete': Integer grid square
                'room': Integer room number
                'cont_discrete': (y,x) position
                'hansen/hansen8': Integer adjacent grid squares
                'grid': Integer grid squares (mxm)
                TODO: 'lidar': [bins+2,] vector of range to nearest wall, then 2D for relative xy position of goal
            obs_m: Range of observation (m) (or size of grid). If wall/goal is out of range, that bin will be 0

            action_failure_probability: Likelihood that taking one action fails and chooses another
            action_type: 'ordinal' (8D compass) or 'cardinal' (4D compass) or 'yx' (2D continuous)
            action_std: Standard deviation of action error (sampled from normal distribution)
            agent_xy: Optionally, provide a fixed (x, y) agent location used every reset. Defaults to random
            goal_xy: Optionally, provide a fixed (x, y) goal location used every reset.
                If you give invalid coordinate (e.g., (0,0) is a wall), uses default goal from layouts.
            step_reward: Reward for each step
            wall_reward: Reward for hitting a wall
            goal_reward: Reward for reaching goal
            goal_threshold: Threshold for being in range of goal
            render_mode
        """
        assert layout in LAYOUTS
        self.metadata["name"] += f"__{layout}__{action_type}__{obs_type}"
        grid = np_to_grid(layout_to_np(LAYOUTS[layout]))
        if "b" in layout:
            layout = layout[:-1]  # Remove b for later indexing
        self.grid = grid
        self.gridshape = np.array(grid.shape)
        (
            self.single_observation_space,
            self._get_obs,
        ) = get_observation_space_and_function(obs_type, self.grid, obs_m, cell_size)
        self.valid_states = np.flatnonzero(
            grid >= 0
        )  # Places where we can put goal or agent
        self.rng, _ = seeding.np_random()
        self.max_velocity = 5.0

        # Different action spaces and random modifiers
        if action_type == "yx":
            self.single_action_space = gymnasium.spaces.Box(-1.0, 1.0, (2,))

            def sample_action(
                a: NDArray[float], rng: np.random.Generator
            ) -> NDArray[float]:
                return a + rng.normal(scale=action_std, size=a.shape)

            self._sample_action = sample_action
        else:
            actions = ACTIONS_CARDINAL if action_type == "cardinal" else ACTIONS_ORDINAL
            action_matrix = create_action_probability_matrix(
                actions.shape[0], action_failure_probability
            )
            self.single_action_space = gymnasium.spaces.Discrete(actions.shape[0])

            def sample_action(
                a: NDArray[int], rng: np.random.Generator
            ) -> NDArray[float]:
                a = vectorized_multinomial_with_rng(action_matrix[a], rng)
                a = actions[a]
                if action_std:
                    return a + rng.normal(scale=action_std, size=a.shape)
                else:
                    return a

            self._sample_action = sample_action
        self.use_velocity = use_velocity
        # Boilerplate for vector environment
        self.num_envs = num_envs
        self.is_vector_env = True
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.observation_space = batch_space(self.single_observation_space, num_envs)

        # Constants
        self.time_limit = time_limit
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.wall_reward = wall_reward
        self.goal_threshold = goal_threshold
        self.cell_size = cell_size
        self.action_power = action_power
        self.render_mode = render_mode

        # Random or fixed goal/agent
        if goal_xy is not None:
            goal_yx = tuple(reversed(goal_xy))  # (x,y) to (y,x)
            if grid[goal_yx] < 0:
                goal_yx = tuple(reversed(ENDS[layout]))
            goal_yx = np.array(goal_yx)
            self._sample_goal = lambda b, rng: grid_to_coord(
                np.full((b, 2), goal_yx, dtype=int)
            )
        else:
            self._sample_goal = lambda b, rng: grid_to_coord(
                np.array(
                    np.unravel_index(rng.choice(self.valid_states, b), self.grid.shape)
                ).swapaxes(0, 1)
            )
        if agent_xy is not None:
            agent_yx = tuple(reversed(agent_xy))
            agent_yx = np.array(agent_yx)
            if grid[agent_yx] < 0:
                agent_yx = tuple(reversed(STARTS[layout]))
            self._sample_agent = lambda b, rng: grid_to_coord(
                np.full((b, 2), agent_yx, dtype=int), cell_size
            )
        else:
            self._sample_agent = lambda b, rng: grid_to_coord(
                np.array(
                    np.unravel_index(rng.choice(self.valid_states, b), self.grid.shape)
                ).swapaxes(0, 1)
            )

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
        if seed is not None:
            self.seed(seed)
        self.elapsed = np.zeros(self.num_envs, int)
        self.goal_yx = self._sample_goal(self.num_envs, self.rng)
        self.agent_yx = self._sample_agent(self.num_envs, self.rng)
        self.agent_yx_velocity = np.zeros((self.num_envs, 2))
        obs = self._get_obs(self.agent_yx, self.goal_yx)
        return obs

    def _reset_some(self, mask: NDArray):
        """Reset only a subset of environments"""
        if b := mask.sum():
            self.elapsed[mask] = 0
            self.goal_yx[mask] = self._sample_goal(b, self.rng)
            self.agent_yx[mask] = self._sample_agent(b, self.rng)
            self.agent_yx_velocity[mask] = 0.0

    def step(
        self, action: ActType
    ) -> Tuple[
        ObsType, NDArray[float], NDArray[bool], NDArray[bool], Union[dict, list]
    ]:
        """Step in environment

        Sample random action failure. Move agent(s) where move is valid.
        Check if we reached goal. Update with step, wall, and goal rewards.
        """
        self.elapsed += 1
        # Movement
        a = self._sample_action(action, self.rng) * self.action_power
        oob = self._apply_action(a)
        # Reward
        r = np.zeros(self.num_envs, dtype=np.float32)
        d = np.linalg.norm(self.agent_yx - self.goal_yx, 2, -1) <= self.goal_threshold
        r += self.step_reward
        r[oob] = self.wall_reward
        r[d] = self.goal_reward
        truncated = self.elapsed > self.time_limit
        self._reset_some(d | truncated)
        return self._get_obs(self.agent_yx, self.goal_yx), r, d, truncated, {}

    def _apply_action(self, randomized_a_yx: NDArray) -> np.ndarray:
        """Actually apply action. Accounts for velocity/position

        If we attempt to enter a wall square, set velocity to 0 and sample a random point in current square"""
        if self.use_velocity:
            self.agent_yx_velocity += randomized_a_yx
            self.agent_yx_velocity.clip(
                -self.max_velocity, self.max_velocity, self.agent_yx_velocity
            )
            proposed_yx = self.agent_yx + self.agent_yx_velocity
        else:
            proposed_yx = self.agent_yx + randomized_a_yx
        proposed_yx = proposed_yx.clip(
            0, self.gridshape - 1 - 1e-6
        )  # Make sure we're still in grid
        oob = self._out_of_bounds(proposed_yx)
        self.agent_yx[~oob] = proposed_yx[~oob]  # Valid actions
        if (
            oob.any()
        ):  # TODO: Put agent in nearest valid grid square to proposed_yx, remove velocity
            inv_ayx = grid_to_coord(
                coord_to_grid(self.agent_yx[oob], self.cell_size), self.cell_size
            )  # Invalid coordinates, resample such that agent stays in current square
            self.agent_yx[oob] = np.clip(
                inv_ayx + self.rng.normal(scale=0.5, size=inv_ayx.shape),
                inv_ayx - self.cell_size / 2,
                inv_ayx + self.cell_size / 2 - 1e-8,
            )
            self.agent_yx_velocity[
                oob
            ] = 0.0  # This is why we compute oob, so we can reset velocity where needed
        return oob

    def _out_of_bounds(self, proposed_yx: NDArray):
        """Return whether given coordinates correspond to empty/goal square"""
        pyx = coord_to_grid(
            proposed_yx, self.cell_size
        )  # Continuous actions might take us out of grid if we're going super fast
        return self.grid[tuple(pyx.T)] == -1
