__all__ = ['CarVecEnv', 'DiscreteActionCarVecEnv']
import gym
import numpy as np
from gym import spaces
from gym.envs.classic_control import rendering as visualize
from gym.utils import seeding
from gym.vector.utils import batch_space

class CarVecEnv(gym.Env):
    # Physics and task params
    MAX_POS = 1.1
    MIN_POS = -MAX_POS
    MAX_SPEED = 0.07
    MIN_ACT = -1.
    MAX_ACT = 1.
    PRIEST = 0.5
    PRIEST_THRESHOLD = 0.2
    POWER = 0.0015

    # Rendering
    SCREEN_WIDTH = 600
    SCREEN_HEIGHT = 400
    SCALE = SCREEN_WIDTH / (MAX_POS - MIN_POS)
    HEIGHT = 0.55

    def __init__(
        self,
        num_envs: int,
        time_limit: int = 160,
        seed=0,
        rendering=False,
    ):
        self.num_envs = num_envs
        self.is_vector_env = True
        self.single_observation_space = spaces.Box(
            np.array([self.MIN_POS, -self.MAX_SPEED, -1.]),
            np.array([self.MAX_POS, self.MAX_SPEED, 1.]),
            dtype=np.float32
        )
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.single_action_space = spaces.Box(self.MIN_ACT, self.MAX_ACT, (1,), dtype=np.float32)
        self.action_space = batch_space(self.single_action_space, num_envs)

        self.setup_view = False
        self.viewer = None
        self.show = rendering

        self.seed(seed)
        self.s = np.zeros((self.num_envs, 3), dtype=np.float32)
        self.time_limit = time_limit
        self.elapsed = np.zeros(self.num_envs, dtype=int)
        self.heavens = np.ones(self.num_envs, dtype=np.float32)
        self.hells = -self.heavens

    def seed(self, seed=None):
        self.rng, seed = seeding.np_random(seed)
        return seed

    def reset(self):
        self._reset_mask(np.ones(self.num_envs, dtype=bool))
        return self._obs()

    def _reset_mask(self, mask):
        b = mask.sum()
        if b:
            self.s[mask] = np.concatenate((self.rng.uniform(-0.2, 0.2, (b, 1)), np.zeros((b,2), dtype=np.float32)),axis=-1)
            self.elapsed[mask] = 0
            self.heavens[mask] = self.rng.choice([-1, 1], b)
            self.hells[mask] = -self.heavens[mask]
            if mask[0] & (self.viewer is not None): self._draw_flags()  # Redraw flags if we sampled 0 index

    def step(self, actions: np.ndarray):
        self.elapsed += 1
        actions = actions.flatten()
        force = np.clip(actions, self.MIN_ACT, self.MAX_ACT)
        # Position, velocity, priest indicator
        new_velocity = np.clip(self.s[:, 1] + (force * self.POWER), -self.MAX_SPEED, self.MAX_SPEED)
        new_position = np.clip(self.s[:, 0] + new_velocity, self.MIN_POS, self.MAX_POS)
        new_velocity[(new_position == self.MIN_POS) & (new_velocity < 0)] = 0
        dones = np.abs(new_position) >= 1.  # Heaven and hell are -1,1
        hh = np.sign(new_position)  # Convert position to heaven/hell
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        rewards[(hh == self.heavens) & dones] = 1.
        rewards[(hh == self.hells) & dones] = -1.
        dones |= (self.elapsed >= self.time_limit)
        directions = np.where((new_position >= self.PRIEST - self.PRIEST_THRESHOLD) &
                               (new_position <= self.PRIEST + self.PRIEST_THRESHOLD), 1., 0.)
        directions[directions == 1] = self.heavens[directions == 1]
        self.s[~dones] = np.column_stack((new_position[~dones], new_velocity[~dones], directions[~dones]))
        self._reset_mask(dones)
        if self.show:
            self.render()

        return self._obs(), rewards, dones, [{}] * self.num_envs

    def _obs(self):
        return self.s

    def render(self, mode='human'):
        self._setup_view()

        pos = self.s[0, 0]
        self.cartrans.set_translation(
            (pos - self.MIN_POS) * self.SCALE,
            self.HEIGHT * self.SCALE,
        )

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


    def _draw_boundary(self):
        flagx = (
            self.PRIEST - self.PRIEST_THRESHOLD - self.MIN_POS
        ) * self.SCALE
        flagy1 = self.HEIGHT * self.SCALE
        flagy2 = flagy1 + 50
        flagpole = visualize.Line((flagx, flagy1), (flagx, flagy2))
        self.viewer.add_geom(flagpole)

        flagx = (
            self.PRIEST + self.PRIEST_THRESHOLD - self.MIN_POS
        ) * self.SCALE
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
        flag.set_color(0., 1., 0.)
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
        flagx = (self.PRIEST - self.MIN_POS) * self.SCALE
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
            self.viewer = visualize.Viewer(
                self.SCREEN_WIDTH, self.SCREEN_HEIGHT
            )
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
    """"""
    def __init__(self, num_actions: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._actions = np.linspace(self.MIN_ACT, self.MAX_ACT, num_actions)
        self.single_action_space = spaces.Discrete(num_actions)
        self.action_space = batch_space(self.single_action_space, self.num_envs)

    def step(self, actions: np.ndarray):
        actions = self._actions[actions]
        return super().step(actions)
