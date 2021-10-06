import numpy as np
import gym
from gym.vector.utils import batch_space
from utils import *
from collections import deque


class TaxiBase(gym.Env):
    """Base for all taxi environments"""
    metadata = {'render.modes': ['human']}
    # grid = TAXI_ROOMS_LAYOUT
    grid_flat = TAXI_ROOMS_LAYOUT.ravel()  # Flattened grid
    spawns = np.where(grid_flat > 1)[0]  # Passenger/goal spawn locations. Must be different
    taxi_spawns = np.flatnonzero(grid_flat)  # Taxi spawn locations. Can spawn at goal/passenger
    shape = BASE_SHAPE

    def __init__(self, num_envs: int = 128, observability: str = 'loc', time_limit: int = 100,
                 reward_wrong_movement: float = -0.1,
                 gamma: float = 0.95, seed=None):
        self.num_envs = num_envs
        self.seed(seed)
        self.time_limit = time_limit
        self.time_elapsed = np.zeros(self.num_envs, dtype=int)
        self.passenger_locations = np.zeros(self.num_envs, dtype=int)  # Where is passenger to start
        self.passenger_in_taxi = np.zeros(self.num_envs, dtype=bool)  # Is passenger in taxi
        self.dropoff_locations = np.zeros(self.num_envs, dtype=int)  # Where is goal
        self.agent_locations = np.zeros(self.num_envs, dtype=int)  # Where is agent
        self.episode_returns = np.zeros(self.num_envs)
        self.reward_per_timestep = -0.1
        self.reward_goal = 2.
        self.reward_wrong_movement = reward_wrong_movement
        self._cur_gamma = np.ones(self.num_envs)
        self.gamma = gamma
        if observability == 'loc':
            self.single_observation_space = gym.spaces.Discrete(self.grid_flat.size)
            self._obs = lambda: self.agent_locations
        elif observability == 'pas':
            self.single_observation_space = gym.spaces.Discrete(int(self.grid_flat.size * 2))
            self._obs = lambda: self.agent_locations + self.passenger_in_taxi * self.grid_flat.size
        elif observability == 'grid':
            self.single_observation_space = gym.spaces.Box(0, 5, (10,), dtype=np.uint8)
            self._obs = self._grid_obs
        self.observation_space = batch_space(self.single_observation_space, num_envs)


    def seed(self, seed=None):
        seed = np.random.SeedSequence(seed).entropy  # Takes care of none
        self.np_random = np.random.default_rng(seed=seed)
        return seed

    def step(self, actions):
        raise NotImplementedError

    def reset(self):
        self._reset_some(np.ones(self.num_envs, dtype=bool))
        return self._obs()

    def _reset_some(self, mask):
        raise NotImplementedError

    def _obs(self):
        raise NotImplementedError

    def _grid_obs(self):
        v = self.grid_flat[self.agent_locations[:, None] + VIEW_3x3]
        v = np.concatenate((v, self.passenger_in_taxi[:, None]), axis=-1)
        return v

    def can_pickup(self):
        return (self.agent_locations == self.passenger_locations) & ~self.passenger_in_taxi

    def can_dropoff(self):
        return (self.agent_locations == self.dropoff_locations) & self.passenger_in_taxi

    def track_stats(self, mask):
        infos = []
        if mask.any():
            returns = self.episode_returns[mask].tolist()
            lengths = self.time_elapsed[mask].tolist()
            infos = [{'episode': {'r': r, 'l': l}} for r, l in zip(returns, lengths)]
        return infos



class POTaxi(TaxiBase):
    def __init__(self, num_envs: int,
                 observability: str = 'pas',  # 'loc' for location only, 'pas' for location and passenger in taxi or not, 'grid' for egocentric 3x3 obs (flattened)
                 reward_wrong_movement: float = -0.1,
                 time_limit: int = 100,
                 seed=None
                 ):
        super().__init__(num_envs, observability, time_limit, reward_wrong_movement, seed)
        self.actions = FLAT_ACTIONS_BASE
        self.single_action_space = gym.spaces.Discrete(len(FLAT_ACTIONS_BASE))
        self.action_space = batch_space(self.single_action_space, num_envs)

    def _reset_some(self, mask):
        b = mask.sum()
        if b:
            self.time_elapsed[mask] = 0
            ints = self.np_random.integers(0, self.spawns.size, (2, b))
            # Adjust overlaps
            ints[1, ints[0] == ints[1]] = (ints[1, ints[0] == ints[1]]+1) % self.spawns.size
            self.passenger_locations[mask] = self.spawns[ints[0]]
            self.dropoff_locations[mask] = self.spawns[ints[1]]
            self.agent_locations[mask] = self.np_random.choice(self.taxi_spawns, b)
            self.episode_returns[mask] = 0.
            self.passenger_in_taxi[mask] = False

    def step(self, actions):
        self.time_elapsed += 1
        new_loc = self.agent_locations + FLAT_ACTIONS_BASE[actions]
        self.agent_locations[self.grid_flat[new_loc] > 0] = new_loc[self.grid_flat[new_loc] > 0]  # Move to valid squares
        r = np.full(self.num_envs, self.reward_per_timestep)  # Always costs to live
        pick = self.can_pickup(); drop = self.can_dropoff()
        goal_achieved = drop & (actions == 5)  # At dropoff point with passenger, attempted dropoff
        bad_move = ~pick & ~drop & (actions == 5)  # Attempted pickup/dropoff invalid
        r[bad_move] += self.reward_wrong_movement  # Punish bad pickup
        r[goal_achieved] += self.reward_goal
        self.episode_returns += r
        self.passenger_in_taxi[pick & (actions == 5)] = True
        d = goal_achieved | (self.time_elapsed >= self.time_limit)
        info = self.track_stats(d)
        self._reset_some(d)
        return self._obs(), r, d, info

class POTaxi_Hard(TaxiBase):
    def __init__(self, num_envs: int,
                 observability: str = 'loc',  # 'loc' for location only, 'pas' for location and passenger in taxi or not, 'grid' for egocentric 3x3 obs (flattened)
                 reward_wrong_movement: float = -0.1,
                 time_limit: int = 100,
                 seed=None
                 ):
        super().__init__(num_envs, observability, time_limit, reward_wrong_movement, seed)
        self.single_action_space = gym.spaces.Discrete(5)  # No-op, turn left, turn right, move forward, pickup/dropoff
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.agent_orientations = np.zeros(self.num_envs, dtype=int)  # Where is agent facing

    def _reset_some(self, mask):
        b = mask.sum()
        if b:
            self.time_elapsed[mask] = 0
            ints = self.np_random.integers(0, self.spawns.size, (2, b))
            # Adjust overlaps
            ints[1, ints[0] == ints[1]] = (ints[1, ints[0] == ints[1]]+1) % self.spawns.size
            self.passenger_locations[mask] = self.spawns[ints[0]]
            self.dropoff_locations[mask] = self.spawns[ints[1]]
            self.agent_locations[mask] = self.np_random.choice(self.taxi_spawns, b)
            self.agent_orientations[mask] = self.np_random.integers(0, 4, b)  # Random facing location, N, E, S, W
            self.episode_returns[mask] = 0.
            self.passenger_in_taxi[mask] = False

    def step(self, actions):
        self.time_elapsed += 1
        new_loc = self.agent_locations + FORWARD_ACTIONS[self.agent_orientations] * (actions == 3)  # If agent moved forward
        self.agent_locations[self.grid_flat[new_loc] > 0] = new_loc[self.grid_flat[new_loc] > 0]  # Move to valid squares
        self.agent_orientations[actions == 1] = (self.agent_orientations[actions == 1] + 1) % 4  # Clockwise
        self.agent_orientations[actions == 2] = (self.agent_orientations[actions == 2] - 1) % 4  # Counterclockwise
        r = np.full(self.num_envs, self.reward_per_timestep)  # Always costs to live
        pick = self.can_pickup(); drop = self.can_dropoff()
        goal_achieved = drop & (actions == 4)  # At dropoff point with passenger, attempted dropoff
        bad_move = ~pick & ~drop & (actions == 4)  # Attempted pickup/dropoff invalid
        r[bad_move] += self.reward_wrong_movement  # Punish bad pickup
        r[goal_achieved] += self.reward_goal
        self.episode_returns += r  # Add to recorded returns
        self.passenger_in_taxi[pick & (actions == 4)] = True
        d = goal_achieved | (self.time_elapsed >= self.time_limit)
        info = self.track_stats(d)
        self._reset_some(d)
        return self._obs(), r, d, info





if __name__ == "__main__":
    e = POTaxi_Hard(16)
    o = e.reset()
    for t in range(10000):
        o, r, d, info = e.step(e.action_space.sample())
        print(info)
    #
    # e2 = POTaxi_Hard(16)
    # o = e.reset()
    # o2 = e2.reset()
    # print(o)
    # o, r, d, info = e.step(e.action_space.sample())
