from typing import Optional

import numpy as np
from gym import spaces
from pathlib import Path
from gym.utils import seeding, EzPickle
from .mujoco_env import MujocoEnv


class AntTagEnv(MujocoEnv, EzPickle):
    def __init__(self,
                 xml_file: str = "ant_tag_small.xml",
                 frame_skip: int = 15,
                 seed: Optional[int] = None
                 ):
        EzPickle.__init__(**locals())
        initial_joint_ranges = np.stack([[-4.5, -4.5, 0.55, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0] + ([0.] * 14)  for _ in range(2)], axis=1)
        initial_joint_ranges[:2, 1] = np.array([4.5, 4.5])  # x,y positions of ant
        self._init_state_space = initial_joint_ranges

        self.name = "AntTag"
        self.seed(seed)
        # Set inital state and goal state spaces
        self.cage_max_xy = np.full(2, 4.5)
        self.visible_radius = 3.0
        self.tag_radius = 1.5
        self.min_distance = 5.0
        self.target_step = 0.5
        MujocoEnv.__init__(self, xml_file, frame_skip)

    def _get_obs(self, target_pos_visible):
        obs = np.concatenate((self.sim.data.qpos, self.sim.data.qvel, np.zeros(2)))
        if target_pos_visible: obs[-2:] = self.sim.data.mocap_pos[0][:2]
        return obs

    def reset_model(self):
        """Reset ant, sample a target position at least min_distance away, add spheres along ant"""
        qp_qv = self.np_random.uniform(self._init_state_space[:, 0], self._init_state_space[:, 1])
        self.set_state(qp_qv[:len(self.init_qpos)], qp_qv[len(self.init_qpos):])
        target_pos = self.np_random.uniform(self._init_state_space[:2, 0], self._init_state_space[:2, 1])
        while np.linalg.norm(qp_qv[:2] - target_pos) <= self.min_distance: target_pos = self.np_random.uniform(self._init_state_space[:2, 0], self._init_state_space[:2, 1])
        self.sim.data.mocap_pos[0, :2] = target_pos
        self.sim.data.mocap_pos[1:3, :2] = qp_qv[:2]
        return self._get_obs(False)

    def _move_target(self, ant_pos, current_target_pos):
        """Move target according to ant position, 1 of [away, 2 orthogonal, stay still]"""
        target2ant_vec = ant_pos - current_target_pos
        target2ant_vec = target2ant_vec / np.linalg.norm(target2ant_vec)
        choose = self.np_random.randint(4) # Move orthogonally, away, or not at all
        vec = np.zeros(2)
        if choose == 0: vec[:] = -target2ant_vec  # Away
        elif choose == 1:
            vec[:] = target2ant_vec[::-1]; vec[-1] *= -1
        elif choose == 2:
            vec[:] = target2ant_vec[::-1]; vec[0] *= -1
        vec *= self.target_step
        vec += current_target_pos
        if (np.abs(vec) > self.cage_max_xy).any(): vec[:] = current_target_pos
        self.sim.data.mocap_pos[0, :2] = vec

    def _do_reveal_target(self):
        ant_pos = self.sim.data.qpos[:2]
        target_pos = self.sim.data.mocap_pos[0, :2]

        d2target = np.linalg.norm(ant_pos - target_pos)
        if (d2target < self.visible_radius):
            reveal_target_pos = True
        else:
            reveal_target_pos = False

        return reveal_target_pos

    # Execute low-level action for number of frames specified by num_frames_skip
    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        ant_pos = self.sim.data.qpos[:2]
        self._move_target(ant_pos, self.sim.data.mocap_pos[0, :2])
        # Move 2 spheres along the ant
        self.sim.data.mocap_pos[1:3, :2] = ant_pos
        done = False
        env_reward = -1
        # + reward and terminate the episode if can tag the target
        d2target = np.linalg.norm(ant_pos - self.sim.data.mocap_pos[0, :2])
        if (d2target <= self.tag_radius):
            env_reward = 0
            done = True

        return self._get_obs(d2target < self.visible_radius), env_reward, done, {}


    def seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        return seed_
