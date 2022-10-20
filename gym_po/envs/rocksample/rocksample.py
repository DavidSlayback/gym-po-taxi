from enum import IntEnum
from typing import Tuple, Sequence, Optional

import gymnasium
from gymnasium.core import ActType, ObsType


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


class RockSample(gymnasium.Env):
    def __init__(
        self,
        num_envs: int,
        map_size: Sequence[int] = (5, 5),
        init_pos: Sequence[int] = (1, 1),
        render_mode: Optional[str] = None,
    ):
        ...

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        ...

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        super().reset(seed=seed, options=options)
