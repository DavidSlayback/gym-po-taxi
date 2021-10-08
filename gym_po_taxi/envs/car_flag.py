import time

import gym
import numpy as np
from gym import spaces
from gym.envs.classic_control import rendering as visualize
from gym.utils import seeding
from gym.vector.utils import batch_space

class CarEnv(gym.Env):
    MAX_POS = 1.1
    MIN_POS = -MAX_POS
    MAX_SPEED = 0.07
    MIN_ACT = -1.
    MAX_ACT = 1.
    PRIEST = 0.5
    POWER = 0.0015

    SCREEN_WIDTH = 600
    SCREEN_HEIGHT = 400
    PRIEST_THRESHOLD = 0.2
    SCALE = SCREEN_WIDTH / (MAX_POS - MIN_POS)
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
            self.s[mask] = np.concatenate((self.rng.uniform(-0.2, 0.2, (b, 1)), np.zeros(b, dtype=np.float32)),axis=-1)
            self.elapsed[mask] = 0
            self.heavens[mask] = self.rng.choice([-1, 1], b)
            self.hells[mask] = -self.heavens[mask]

    def step(self, actions: np.ndarray):
        self.elapsed += 1
        force = np.clip(actions, self.MIN_ACT, self.MAX_ACT)
        # Position, velocity, priest indicator
        position = self._state[0]
        velocity = self._state[1]
        new_velocity = np.clip(self.s[:, 1] + (force * self.POWER), -self.MAX_SPEED, self.MAX_SPEED)
        new_position = np.clip(self.s[:, 0] + new_velocity, -self.MIN_POS, self.MAX_POS)
        new_velocity[(new_position == self.MIN_POS) & (new_velocity < 0)] = 0
        dones = np.abs(new_position) >= 1.  # Heaven and hell are -1,1
        rewards = np.zeros(self.num_envs)
        rewards[]
        reward = 0.0
        if self.heaven_position > self.hell_position:
            if position >= self.heaven_position:
                reward = 1.0

            if position <= self.hell_position:
                reward = -1.0
                # env_reward = self.steps_cnt - self.max_ep_length

        if self.heaven_position < self.hell_position:
            if position <= self.heaven_position:
                reward = 1.0

            if position >= self.hell_position:
                reward = -1.0
                # env_reward = self.steps_cnt - self.max_ep_length

        direction = 0.0
        if (
            position >= self.priest_position - self.priest_delta
            and position <= self.priest_position + self.priest_delta
        ):
            if self.heaven_position > self.hell_position:
                # Heaven on the right
                direction = 1.0
            else:
                # Heaven on the left
                direction = -1.0

        self._state = np.array([position, velocity, direction])
        self.solved = reward > 0.0

        # if self.solved:
        #     env_reward = 0

        if self.show:
            self.render()

        # return self._state, env_reward, done, {"is_success": reward > 0.0}
        return self._state, reward, done, {"is_success": reward > 0.0}

    def render(self, mode='human'):
        self._setup_view()

        pos = self._state[0]
        self.cartrans.set_translation(
            (pos - self.min_position) * self.scale,
            self._height(pos) * self.scale,
        )

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def reset(self):

        self.solved = False
        self.done = False
        self.steps_cnt = 0

        # Randomize the heaven/hell location
        if self.np_random.randint(2) == 0:
            self.heaven_position = 1.0
        else:
            self.heaven_position = -1.0

        self.hell_position = -self.heaven_position

        if self.viewer is not None:
            self._draw_flags()
            self._draw_boundary()

        self._state = np.array(
            [self.np_random.uniform(low=-0.2, high=0.2), 0, 0.0]
        )
        return np.array(self._state)

    def _height(self, xs):
        return 0.55 * np.ones_like(xs)

    def _draw_boundary(self):
        flagx = (
            self.priest_position - self.priest_delta - self.min_position
        ) * self.scale
        flagy1 = self._height(self.priest_position) * self.scale
        flagy2 = flagy1 + 50
        flagpole = visualize.Line((flagx, flagy1), (flagx, flagy2))
        self.viewer.add_geom(flagpole)

        flagx = (
            self.priest_position + self.priest_delta - self.min_position
        ) * self.scale
        flagy1 = self._height(self.priest_position) * self.scale
        flagy2 = flagy1 + 50
        flagpole = visualize.Line((flagx, flagy1), (flagx, flagy2))
        self.viewer.add_geom(flagpole)

    def _draw_flags(self):
        scale = self.scale
        # Flag Heaven
        flagx = (abs(self.heaven_position) - self.min_position) * scale
        flagy1 = self._height(self.heaven_position) * scale
        flagy2 = flagy1 + 50
        flagpole = visualize.Line((flagx, flagy1), (flagx, flagy2))
        self.viewer.add_geom(flagpole)
        flag = visualize.FilledPolygon(
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
        )

        # RED for hell
        if self.heaven_position > self.hell_position:
            flag.set_color(0.0, 1.0, 0)
        else:
            flag.set_color(1.0, 0.0, 0)

        self.viewer.add_geom(flag)

        # Flag Hell
        flagx = (-abs(self.heaven_position) - self.min_position) * scale
        flagy1 = self._height(self.hell_position) * scale
        flagy2 = flagy1 + 50
        flagpole = visualize.Line((flagx, flagy1), (flagx, flagy2))
        self.viewer.add_geom(flagpole)
        flag = visualize.FilledPolygon(
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
        )

        # GREEN for heaven
        if self.heaven_position > self.hell_position:
            flag.set_color(1.0, 0.0, 0)
        else:
            flag.set_color(0.0, 1.0, 0)

        self.viewer.add_geom(flag)

        # BLUE for priest
        flagx = (self.priest_position - self.min_position) * scale
        flagy1 = self._height(self.priest_position) * scale
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
                self.screen_width, self.screen_height
            )
            scale = self.scale
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs - self.min_position) * scale, ys * scale))

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

            if self.args is not None:
                if self.n_layers in [2, 3]:

                    ################ Goal 1 ################
                    car1 = visualize.FilledPolygon(
                        [(l, b), (l, t), (r, t), (r, b)]
                    )
                    car1.set_color(1, 0.0, 0.0)
                    car1.add_attr(
                        visualize.Transform(translation=(0, clearance))
                    )
                    self.cartrans1 = visualize.Transform()
                    car1.add_attr(self.cartrans1)
                    self.viewer.add_geom(car1)
                    ######################################

                if self.n_layers in [3]:

                    ############### Goal 2 ###############
                    car2 = visualize.FilledPolygon(
                        [(l, b), (l, t), (r, t), (r, b)]
                    )
                    car2.set_color(0.0, 1, 0.0)
                    car2.add_attr(
                        visualize.Transform(translation=(0, clearance))
                    )
                    self.cartrans2 = visualize.Transform()
                    car2.add_attr(self.cartrans2)
                    self.viewer.add_geom(car2)
                    ######################################

            self.setup_view = True

    def display_subgoals(self, subgoals, mode="human"):
        self._setup_view()

        if self.show:
            pos = self._state[0]
            self.cartrans.set_translation(
                (pos - self.min_position) * self.scale,
                self._height(pos) * self.scale,
            )

            if self.n_layers in [2, 3]:
                pos1 = subgoals[0][0]
                self.cartrans1.set_translation(
                    (pos1 - self.min_position) * self.scale,
                    self._height(pos1) * self.scale,
                )

            if self.n_layers in [3]:
                pos2 = subgoals[1][0]
                self.cartrans2.set_translation(
                    (pos2 - self.min_position) * self.scale,
                    self._height(pos2) * self.scale,
                )

            return self.viewer.render(return_rgb_array=mode == 'rgb_array')
        else:
            return

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class CarEnvWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, *, num_actions: int):
        super().__init__(env)

        self.state_space = gym.spaces.Box(
            low=self.low_state, high=self.high_state
        )
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