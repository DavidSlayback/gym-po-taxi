from os import path
from typing import Tuple

import gymnasium
import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.utils import EzPickle

GREEN = [0, 1, 0, 0.5]
RED = [1, 0, 0, 0.5]


class AntHeavenHellEnv(MujocoEnv, EzPickle):
    """HeavenHell, but for an "ant" robot

    Ant starts at bottom of T-maze. Navigates to priest located in center of top of T. While in range of priest, ant
    can observe which direction is heaven or hell
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 3,
    }

    def __init__(
        self,
        xml_file: str = path.join(
            path.dirname(__file__), "assets", "ant_heaven_hell.xml"
        ),
        frame_skip: int = 15,
        heaven_hell: Tuple[Tuple[float, float], Tuple[float, float]] = (
            (-6.25, 6.0),
            (6.25, 6.0),
        ),
        priest_pos: Tuple[float, float] = (0.0, 6.0),
        termination_radius: float = 2.0,
        **kwargs,
    ):
        EzPickle.__init__(**locals())
        self._hhp = np.stack(heaven_hell + (priest_pos,))
        self._r = termination_radius
        self.heaven_pos = self._hhp[0]
        self.hell_pos = self._hhp[1]
        self.heaven_direction = np.sign(self.heaven_pos[0])

        initial_joint_ranges = np.stack(
            [
                [
                    -1.0,
                    0,
                    0.55,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    -1.0,
                    0.0,
                    -1.0,
                    0.0,
                    1.0,
                ]
                + ([0.0] * 14)
                for _ in range(2)
            ],
            axis=1,
        )
        initial_joint_ranges[:2, 1] = np.array([1.0, 1.0])  # x,y positions of ant
        self._init_state_space = initial_joint_ranges

        self.name = "AntHeavenHell"
        obs_shape = 28  # Ant, without current position, + 1D for heaven
        observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32
        )
        MujocoEnv.__init__(
            self, xml_file, frame_skip, observation_space, **kwargs
        )  # Also handles action space
        self.left_name = "left_area"
        self.right_name = "right_area"

    def _get_obs(self, reveal_heaven_direction):
        """Get standard ant observation, plus either 0 (if not in range of priest) or +-1 (heaven position)"""
        position = self.data.qpos.flat.copy()[
            2:
        ]  # Exclude xy position from observation
        velocity = self.data.qvel.flat.copy()
        obs = np.concatenate((position, velocity, np.zeros(1)))
        if reveal_heaven_direction:
            obs[-1] = self.heaven_direction
        return obs

    def reset_model(self):
        """Reset ant position. Randomly sample heaven and hell"""
        qp_qv = self.np_random.uniform(
            self._init_state_space[:, 0], self._init_state_space[:, 1]
        )
        self.set_state(qp_qv[: len(self.init_qpos)], qp_qv[len(self.init_qpos) :])
        # -1: heaven on left, 1: heaven on the right
        flip = int(self.np_random.uniform() >= 0.5)
        self.heaven_pos = self._hhp[flip]
        self.hell_pos = self._hhp[1 - flip]
        self.heaven_direction = np.sign(self.heaven_pos[0])
        # Changing the color of heaven/hell areas
        if self.heaven_direction > 0:
            # heaven on the right
            self.model.site(self.right_name).rgba = GREEN
            self.model.site(self.left_name).rgba = RED
        else:
            # heaven on the left
            self.model.site(self.right_name).rgba = RED
            self.model.site(self.left_name).rgba = GREEN
        return self._get_obs(False)

    def step(self, action):
        """Step using action."""
        self.do_simulation(action, self.frame_skip)
        # Distances from heaven, hell, and priest
        distances = np.linalg.norm(self.data.qpos[:2] - self._hhp, axis=-1)
        done = (distances[:2] <= self._r).any()  # Are we in radius of heaven/hell
        priest_in_range = distances[2] <= self._r  # Can we see priest?
        heaven_dist = distances[int(max(self.heaven_direction, 0))]
        if done:  # +-1 reward
            if heaven_dist <= self._r:
                r = 1.0
            else:
                r = -1.0
        else:
            r = 0.0

        return self._get_obs(priest_in_range), r, done, False, {}
