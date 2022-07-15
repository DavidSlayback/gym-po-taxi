from typing import Tuple, Sequence, Optional, Union

import gym
import numpy as np
from enum import IntEnum
import gym
from gym.core import ActType, ObsType


class Obs(IntEnum):
    NULL = 0
    GOOD = 1
    BAD = 2

class ACTION(IntEnum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3
    SAMPLE = 4


class RockSample(gym.Env):
    def __init__(self, num_envs: int, map_size: Sequence[int] = (5, 5), init_pos: Sequence[int] = (1, 1)):
        ...

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        ...

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[ObsType, tuple[ObsType, dict]]:
        ...

    def seed(self, seed=None):
        self.rng, seed = seed