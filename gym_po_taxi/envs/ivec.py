from gym import Env, Wrapper
from gym.vector import VectorEnvWrapper
import numpy as np
"""Subclass basic wrappers to avoid some of the hassles"""
class IVecTimeLimit(Wrapper):
    """Time limit, track steps for all envs"""
    def __init__(self, env, time_limit: int):
        super().__init__(env)
        assert time_limit > 0
        self.time_limit = time_limit
        self.time_elapsed = np.zeros(env.num_envs)

    def reset(self):
        self.time_elapsed[:] = 0
        return super().reset()

    def step(self, actions):
        self.time_elapsed[:] += 1
        o, r, d, info = super().step(actions)
        d[self.time_elapsed > self.time_limit] = True
        return o, r, d, info

