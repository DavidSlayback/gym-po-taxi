import time

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
        args=None,
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

        #################### START CONFIGS #######################
        self.args = args

        self.action_dim = 1
        self.action_bounds = [1.0]
        self.action_offset = np.zeros((len(self.action_bounds)))

        self.subgoal_bounds = np.array(
            [
                [self.MIN_POS, self.MAX_POS],
                [-self.MAX_SPEED, self.MAX_SPEED],
            ]
        )
        self.subgoal_dim = len(self.subgoal_bounds)

        # functions to project state to goal
        self.project_state_to_subgoal = lambda sim, state: state[:-1]
        self.subgoal_bounds_symmetric = np.zeros((len(self.subgoal_bounds)))
        self.subgoal_bounds_offset = np.zeros((len(self.subgoal_bounds)))

        for i in range(len(self.subgoal_bounds)):
            self.subgoal_bounds_symmetric[i] = (
                self.subgoal_bounds[i][1] - self.subgoal_bounds[i][0]
            ) / 2
            self.subgoal_bounds_offset[i] = (
                self.subgoal_bounds[i][1] - self.subgoal_bounds_symmetric[i]
            )

        self.subgoal_thresholds = np.array([0.05, 0.01])

        self.state_dim = 3
        self.low_obs_dim = 2

        self.name = "Car-Flag-POMDP"

        # Configs for agent
        agent_params = {}
        agent_params["subgoal_test_perc"] = 0.3

        agent_params["random_action_perc"] = 0.2
        agent_params["num_pre_training_episodes"] = -1

        agent_params["atomic_noise"] = [0.1]
        agent_params["subgoal_noise"] = [0.1, 0.1]

        agent_params["num_exploration_episodes"] = 50

        self.agent_params = agent_params
        self.sim = None
        #################### END CONFIGS #######################
        self.setup_view = False
        self.viewer = None
        self.show = rendering

        if args is not None:
            self.n_layers = args.n_layers

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
        dones |= (self.elapsed > self.time_limit)
        directions = np.where((new_position >= self.PRIEST - self.PRIEST_THRESHOLD) &
                               (new_position <= self.PRIEST + self.PRIEST_THRESHOLD), 1., 0.)
        directions[directions == 1] = self.heavens[directions == 1]
        self.s[~dones] = np.column_stack((new_position[~dones], new_velocity[~dones], directions[~dones]))
        self._reset_mask(dones)
        if self.show:
            self.render()

        return self._obs(), rewards, dones, {"is_success": rewards > 0.0}

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



class CarEnvWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, *, num_actions: int):
        super().__init__(env)

        self.action_space = gym.spaces.Discrete(num_actions)
        self.__actions = np.linspace(
            self.min_action, self.max_action, num_actions
        )

    def action(self, action):
        return self.__actions[action]

    def reverse_action(self, action):
        return next(i for i, a in enumerate(self.__actions) if a == action)

    @property
    def state(self):
        state = self.env._state.copy()
        state[-1] = self.env.heaven_position
        return state
