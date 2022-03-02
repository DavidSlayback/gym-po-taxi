from typing import Tuple, Optional
import numpy as np
from gym.utils import seeding, EzPickle
from .mujoco_env import MujocoEnv

GREEN = [0, 1, 0, 0.5]
RED = [1, 0, 0, 0.5]

class AntHeavenHellEnv(MujocoEnv, EzPickle):
    def __init__(self,
                 xml_file: str = "ant_heaven_hell.xml",
                 frame_skip: int = 15,
                 heaven_hell: Tuple[Tuple[float, float], Tuple[float, float]] = ((-6.25, 6.0), (6.25, 6.0)),
                 priest_pos: Tuple[float, float] = (0., 6.),
                 termination_radius: float = 2.,
                 seed: Optional[int] = None
                 ):

        #################### START CONFIGS #######################

        EzPickle.__init__(**locals())
        self._hhp = np.stack(heaven_hell + (priest_pos,))
        self._r = termination_radius
        self.heaven_pos = self._hhp[0]
        self.hell_pos = self._hhp[1]
        self.heaven_direction = np.sign(self.heaven_pos[0])

        initial_joint_ranges = np.stack([[-1., 0, 0.55, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0] + ([0.] * 14)  for _ in range(2)], axis=1)
        initial_joint_ranges[:2, 1] = np.array([1., 1.])  # x,y positions of ant
        self._init_state_space = initial_joint_ranges

        self.name = "AntHeavenHell"

        MujocoEnv.__init__(self, xml_file, frame_skip)
        self.left_id = self.model.name2id('left_area', 'site')
        self.right_id = self.model.name2id('right_area', 'site')
        self.seed(seed)

    def _get_obs(self, reveal_heaven_direction):
        obs = np.concatenate((self.sim.data.qpos, self.sim.data.qvel, np.zeros(1)))
        if reveal_heaven_direction: obs[-1] = self.heaven_direction
        return obs

    # Reset simulation to state within initial state specified by user
    def reset_model(self):
        qp_qv = self.np_random.uniform(self._init_state_space[:, 0], self._init_state_space[:, 1])
        self.set_state(qp_qv[:len(self.init_qpos)], qp_qv[len(self.init_qpos):])
        # -1: heaven on left, 1: heaven on the right
        flip = int(self.np_random.rand() >= 0.5)
        self.heaven_pos = self._hhp[flip]
        self.hell_pos = self._hhp[1-flip]
        self.heaven_direction = np.sign(self.heaven_pos[0])
        # Changing the color of heaven/hell areas
        if self.heaven_direction > 0:
            # heaven on the right
            self.sim.model.site_rgba[self.right_id] = GREEN
            self.sim.model.site_rgba[self.left_id] = RED
        else:
            # heaven on the left
            self.sim.model.site_rgba[self.left_id] = GREEN
            self.sim.model.site_rgba[self.right_id] = RED
        return self._get_obs(False)

    # Execute low-level action for number of frames specified by num_frames_skip
    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        distances = np.linalg.norm(self.sim.data.qpos[:2] - self._hhp, axis=-1)
        done = (distances[:2] <= self._r).any()
        priest_in_range = distances[2] <= self._r
        heaven_dist = distances[int(max(self.heaven_direction, 0))]
        if done:  # In range of heaven or hell
            if heaven_dist <= self._r: r = 0.
            else: r = -10.
        else: r = -1.

        return self._get_obs(priest_in_range), r, done, {}

    def seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        return seed_
