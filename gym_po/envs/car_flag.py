__all__ = ["CarVecEnv", "DiscreteActionCarVecEnv"]
import os
from typing import Optional, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.core import ObsType
from numpy.typing import NDArray

if os.name != "nt":
    # Windows can't handle headless (missing EGL dll)
    import pyglet

    pyglet.options["headless"] = True
# try:
#     from gym.envs.classic_control import rendering as visualize  # Use pyglet
# except:
visualize = None  # Use number line
from gymnasium.vector.utils import batch_space


class CarVecEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    # Physics and task params
    MAX_POS = 1.1
    MIN_POS = -MAX_POS
    POS_RANGE = MAX_POS - MIN_POS
    MAX_SPEED = 0.07
    MIN_ACT = -1.0
    MAX_ACT = 1.0
    PRIEST = 0.5
    PRIEST_THRESHOLD = 0.2
    POWER = 0.0015

    # Rendering
    SCREEN_WIDTH = 600
    SCREEN_HEIGHT = 400
    SCALE = SCREEN_WIDTH / (MAX_POS - MIN_POS)
    HEIGHT = 0.55
    PIXEL_WIDTH = 4
    PIXEL_HEIGHT = 24
    START_PIXEL_RANGE = 600 - PIXEL_WIDTH
    # START_PIXEL_BINS = np.arange(SCREEN_WIDTH - 4)  # All ints along this range as start idx
    NLINE = np.zeros((SCREEN_WIDTH, PIXEL_HEIGHT * 2, 3), dtype=np.uint8)
    NLINE[0:PIXEL_WIDTH] = 255
    NLINE[-PIXEL_WIDTH:] = 255  # Endpoints are white
    NLINE = NLINE.swapaxes(0, 1)

    def __init__(
        self,
        num_envs: int,
        time_limit: int = 160,
        render_mode: Optional[str] = None,
    ):
        self.num_envs = num_envs
        self.is_vector_env = True
        self.single_observation_space = spaces.Box(
            np.array([self.MIN_POS, -self.MAX_SPEED, -1.0]),
            np.array([self.MAX_POS, self.MAX_SPEED, 1.0]),
            dtype=np.float32,
        )
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.single_action_space = spaces.Box(
            self.MIN_ACT, self.MAX_ACT, (1,), dtype=np.float32
        )
        self.action_space = batch_space(self.single_action_space, num_envs)

        self.setup_view = False
        self.viewer = None
        self.render_mode = render_mode

        self.s = np.zeros((self.num_envs, 3), dtype=np.float32)
        self.time_limit = time_limit
        self.elapsed = np.zeros(self.num_envs, dtype=int)
        self.heavens = np.ones(self.num_envs, dtype=np.float32)
        self.priests = np.full(self.num_envs, self.PRIEST)
        self.hells = -self.heavens

        self.pixel_conversion = lambda x: np.floor(
            np.interp(
                x, xp=[self.MIN_POS, self.MAX_POS], fp=[0, self.START_PIXEL_RANGE]
            )
        ).astype(int)
        self.PIXEL_FLAGS = self.pixel_conversion([-1, 1])

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        super().reset(seed=seed, options=options)
        self._reset_mask(np.ones(self.num_envs, dtype=bool))
        return self._obs(), {}

    def _reset_mask(self, mask):
        b = mask.sum()
        if b:
            self.s[mask] = np.concatenate(
                (
                    self.np_random.uniform(-0.2, 0.2, (b, 1)),
                    np.zeros((b, 2), dtype=np.float32),
                ),
                axis=-1,
            )
            self.elapsed[mask] = 0
            self.heavens[mask] = self.np_random.choice([-1, 1], b)
            self.hells[mask] = -self.heavens[mask]
            self.priests[mask] = self.np_random.choice([-self.PRIEST, self.PRIEST], b)
            if mask[0] & (self.viewer is not None):
                self._draw_flags()  # Redraw flags if we sampled 0 index

    def step(self, actions: NDArray[float]):
        self.elapsed += 1
        actions = actions.flatten()
        force = np.clip(actions, self.MIN_ACT, self.MAX_ACT)
        # Position, velocity, priest indicator
        new_velocity = np.clip(
            self.s[:, 1] + (force * self.POWER), -self.MAX_SPEED, self.MAX_SPEED
        )
        new_position = np.clip(self.s[:, 0] + new_velocity, self.MIN_POS, self.MAX_POS)
        new_velocity[(new_position == self.MIN_POS) & (new_velocity < 0)] = 0
        dones = np.abs(new_position) >= 1.0  # Heaven and hell are -1,1
        hh = np.sign(new_position)  # Convert position to heaven/hell
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        rewards[(hh == self.heavens) & dones] = 1.0
        rewards[(hh == self.hells) & dones] = -1.0
        truncated = self.elapsed >= self.time_limit
        directions = np.where(
            (new_position >= self.priests - self.PRIEST_THRESHOLD)
            & (new_position <= self.priests + self.PRIEST_THRESHOLD),
            self.heavens,
            0.0,
        )
        # directions[directions == 1] = self.heavens[directions == 1]
        self.s[~dones] = np.column_stack(
            (new_position[~dones], new_velocity[~dones], directions[~dones])
        )
        self._reset_mask(dones | truncated)
        return self._obs(), rewards, dones, truncated, {}

    def _obs(self):
        return self.s

    def render(self):
        if visualize is not None:  # Standard classic control rendering
            self._setup_view()
            pos = self.s[0, 0]
            self.cartrans.set_translation(
                (pos - self.MIN_POS) * self.SCALE,
                self.HEIGHT * self.SCALE,
            )
            return self.viewer.render(return_rgb_array=self.render_mode == "rbg_array")
        else:
            pos = self.s[0, 0]  # Get float position
            pixel_pos = self.pixel_conversion(pos)
            priest_poses = self.pixel_conversion(
                [
                    self.priests[0] - self.PRIEST_THRESHOLD,
                    self.priests[0],
                    self.priests[0] + self.PRIEST_THRESHOLD,
                ]
            )
            img = self.NLINE.copy()  # new img
            hea_idx = 0 if self.heavens[0] < 0 else 1
            hell_idx = 1 - hea_idx
            hea_pos, hell_pos = self.PIXEL_FLAGS[hea_idx], self.PIXEL_FLAGS[hell_idx]
            img[:, hea_pos : hea_pos + 4, 1] = 255
            img[:, hell_pos : hell_pos + 4, 0] = 255
            img[-self.PIXEL_HEIGHT :, pixel_pos : pixel_pos + 4] = (
                255 if self.s[0, -1] else 128
            )
            img[:, priest_poses[0] : priest_poses[0] + self.PIXEL_WIDTH, 2] = 128
            img[:, priest_poses[2] : priest_poses[2] + self.PIXEL_WIDTH, 2] = 128
            img[:, priest_poses[1] : priest_poses[1] + self.PIXEL_WIDTH, 2] = 255
            if self.render_mode == "rgb_array":
                return img
            else:
                import pygame

                if self.viewer is None:
                    pygame.init()
                    self.viewer = pygame.display.set_mode(img.shape[:-1])
                sfc = pygame.surfarray.make_surface(img)
                self.viewer.blit(sfc, (0, 0))
                pygame.display.update()
                return img

    def _draw_boundary(self):
        flagx = (self.priests[0] - self.PRIEST_THRESHOLD - self.MIN_POS) * self.SCALE
        flagy1 = self.HEIGHT * self.SCALE
        flagy2 = flagy1 + 50
        flagpole = visualize.Line((flagx, flagy1), (flagx, flagy2))
        self.viewer.add_geom(flagpole)

        flagx = (self.priests[0] + self.PRIEST_THRESHOLD - self.MIN_POS) * self.SCALE
        flagy1 = self.HEIGHT * self.SCALE
        flagy2 = flagy1 + 50
        flagpole = visualize.Line((flagx, flagy1), (flagx, flagy2))
        self.viewer.add_geom(flagpole)

    def _draw_flags(self):
        # Flag Heaven
        flagx = (self.heavens[0] - self.MIN_POS) * self.SCALE
        flagy1 = self.HEIGHT * self.SCALE
        flagy2 = flagy1 + 50
        flagpole = visualize.Line((flagx, flagy1), (flagx, flagy2))
        self.viewer.add_geom(flagpole)
        flag = visualize.FilledPolygon(
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
        )
        flag.set_color(0.0, 1.0, 0.0)
        self.viewer.add_geom(flag)

        # Flag Hell
        flagx = (self.hells[0] - self.MIN_POS) * self.SCALE
        flagy1 = self.HEIGHT * self.SCALE
        flagy2 = flagy1 + 50
        flagpole = visualize.Line((flagx, flagy1), (flagx, flagy2))
        self.viewer.add_geom(flagpole)
        flag = visualize.FilledPolygon(
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
        )
        flag.set_color(1.0, 0.0, 0)
        self.viewer.add_geom(flag)

        # BLUE for priest
        flagx = (self.priests[0] - self.MIN_POS) * self.SCALE
        flagy1 = self.HEIGHT * self.SCALE
        flagy2 = flagy1 + 50
        flagpole = visualize.Line((flagx, flagy1), (flagx, flagy2))
        self.viewer.add_geom(flagpole)
        flag = visualize.FilledPolygon(
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
        )
        flag.set_color(0.0, 0.0, 1.0)
        self.viewer.add_geom(flag)

    def _setup_view(self):
        if not self.setup_view:
            self.viewer = visualize.Viewer(self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
            xs = np.linspace(self.MIN_POS, self.MAX_POS, 100)
            ys = np.full_like(xs, self.HEIGHT)
            xys = list(zip((xs - self.MIN_POS) * self.SCALE, ys * self.SCALE))

            self.track = visualize.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10
            carwidth = 40
            carheight = 20

            l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
            car = visualize.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(visualize.Transform(translation=(0, clearance)))
            self.cartrans = visualize.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = visualize.make_circle(carheight / 2.5)
            frontwheel.set_color(0.5, 0.5, 0.5)
            frontwheel.add_attr(
                visualize.Transform(translation=(carwidth / 4, clearance))
            )
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = visualize.make_circle(carheight / 2.5)
            backwheel.add_attr(
                visualize.Transform(translation=(-carwidth / 4, clearance))
            )
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(0.5, 0.5, 0.5)
            self.viewer.add_geom(backwheel)

            self._draw_flags()
            self._draw_boundary()
            self.setup_view = True

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class DiscreteActionCarVecEnv(CarVecEnv):
    """Discrete action car environment. Evenly spaced actions along control dimension"""

    def __init__(self, num_actions: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._actions = np.linspace(self.MIN_ACT, self.MAX_ACT, num_actions)
        l, r, c, nact = "<", ">", ":", num_actions // 2
        self.action_names = ["<" * i + ":" for i in reversed(range(1, nact + 1))] + [
            ":" + ">" * i for i in range(1, nact + 1)
        ]
        if num_actions % 2 == 1:
            self.action_names.insert(nact, ":")  # Null action
        self.single_action_space = spaces.Discrete(num_actions)
        self.action_space = batch_space(self.single_action_space, self.num_envs)

    def step(self, actions: NDArray[float]):
        actions = self._actions[actions]
        return super().step(actions)
