import time
from collections import deque
import numpy as np
import gym


class RecordEpisodeStatistics(gym.Wrapper):
    """Modification of original gym EpiscodeStatistics to consider discounted return"""
    def __init__(self, env, discount: float, deque_size: int = 100):
        super(RecordEpisodeStatistics, self).__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.t0 = time.perf_counter()
        self.episode_count = 0
        self.episode_returns = None
        self.discounted_episode_returns = None
        self.episode_lengths = None
        self._cur_discount = None
        self.discount = discount or getattr(env, "discount", 0) or 1.
        self.return_queue = deque(maxlen=deque_size)
        self.discounted_return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)
        self.is_vector_env = getattr(env, "is_vector_env", False)

    def reset(self, **kwargs):
        observations = super(RecordEpisodeStatistics, self).reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float64)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.discounted_episode_returns = np.zeros(self.num_envs, dtype=np.float64)
        self._cur_discount = np.ones(self.num_envs, dtype=np.float64)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super(RecordEpisodeStatistics, self).step(
            action
        )
        self.episode_returns += rewards
        self.discounted_episode_returns += rewards * self._cur_discount
        self.episode_lengths += 1
        self._cur_discount *= self.discount
        if not self.is_vector_env:
            infos = [infos]
            dones = [dones]
        d_idx = np.flatnonzero(dones)
        nd = d_idx.size
        infos = list(infos)
        if nd:
            for i in d_idx:
                infos[i] = infos[i].copy()
                episode_info = {
                    "r": self.episode_returns[i],
                    "l": self.episode_lengths[i],
                    "dr": self.discounted_episode_returns[i],
                    "t": round(time.perf_counter() - self.t0, 6),
                }
                infos[i]["episode"] = episode_info
            self.return_queue.extend(self.episode_returns[d_idx])
            self.discounted_return_queue.extend(self.discounted_episode_returns[d_idx])
            self.length_queue.extend(self.episode_lengths[d_idx])
            self.episode_count += nd
            self.episode_returns[d_idx] = 0
            self.episode_lengths[d_idx] = 0
            self.discounted_episode_returns[d_idx] = 0
            self._cur_discount[d_idx] = 1.
        return (
            observations,
            rewards,
            dones if self.is_vector_env else dones[0],
            infos if self.is_vector_env else infos[0],
        )