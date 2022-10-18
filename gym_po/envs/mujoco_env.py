from collections import OrderedDict
from os import path
import sys
from typing import Optional

from gym import error, logger, spaces
import numpy as np
import gym

from gym.envs.mujoco.mujoco_env import MujocoEnv


DEFAULT_SIZE = 480

# Default window dimensions.
DEFAULT_WINDOW_WIDTH = 640
DEFAULT_WINDOW_HEIGHT = 480

# Default window title.
DEFAULT_WINDOW_TITLE = "MuJoCo Viewer"

# Internal renderbuffer size, in pixels.
_MAX_RENDERBUFFER_SIZE = 2048


def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(
            OrderedDict(
                [
                    (key, convert_observation_to_space(value))
                    for key, value in observation.items()
                ]
            )
        )
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float("inf"), dtype=np.float32)
        high = np.full(observation.shape, float("inf"), dtype=np.float32)
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments."""

    def __init__(self, model_path, frame_skip, mujoco_framework="dm_control"):

        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = path.join(path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise OSError(f"File {fullpath} does not exist")
        if mujoco_framework == "mujoco_py":
            logger.warn(
                "This version of the mujoco environments depends "
                "on the mujoco-py bindings, which are no longer maintained "
                "and may stop working. Please upgrade to the v4 versions of "
                "the environments (which depend on dm_control instead), unless "
                "you are trying to precisely replicate previous works)."
            )
            try:
                import mujoco_py

                self._mujoco_framework = mujoco_py
            except ImportError as e:
                raise error.DependencyNotInstalled(
                    "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(
                        e
                    )
                )

            self.model = self._mujoco_framework.load_model_from_path(fullpath)
            self.sim = self._mujoco_framework.MjSim(self.model)
            self._viewers = {}

        else:
            try:
                import dm_control.mujoco as dm_mujoco

                self._mujoco_framework = dm_mujoco
            except ImportError as e:
                raise error.DependencyNotInstalled(
                    "{}. (HINT: you need to install dm_control)".format(e)
                )

            self.sim = self._mujoco_framework.Physics.from_xml_path(fullpath)
            self.model = self.sim.model
            self.util = self._mujoco_framework.wrapper.util
            self.mjlib = self._mujoco_framework.wrapper.mjbindings.mjlib
            self._str2type = self._mujoco_framework.wrapper.core._str2type

        self.frame_skip = frame_skip

        self.data = self.sim.data
        self.viewer = None

        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array"],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        self._set_action_space()

        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(action)
        assert not done

        self._set_observation_space(observation)

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    # -----------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        # super().reset(seed=seed)
        self.sim.reset()
        ob = self.reset_model()
        if not return_info:
            return ob
        else:
            return ob, {}

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        state = self.sim.get_state()
        if self._mujoco_framework.__name__ == "mujoco_py":
            state = self._mujoco_framework.MjSimState(
                state.time, qpos, qvel, state.act, state.udd_state
            )
        else:
            state[: self.model.nq] = qpos
            state[self.model.nq : self.model.nq + self.model.nv] = qvel
        self.sim.set_state(state)
        self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        if np.array(ctrl).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()

        if self._mujoco_framework.__name__ == "dm_control":
            self.mjlib.mj_rnePostConstraint(self.model.ptr, self.sim.data.ptr)

    def render(
        self,
        mode="human",
        width=DEFAULT_SIZE,
        height=DEFAULT_SIZE,
        camera_id=None,
        camera_name=None,
    ):
        if mode == "rgb_array" or mode == "depth_array":
            if camera_id is not None and camera_name is not None:
                raise ValueError(
                    "Both `camera_id` and `camera_name` cannot be"
                    " specified at the same time."
                )

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = "track"

            if camera_id is None:
                if self._mujoco_framework.__name__ == "mujoco_py":
                    if camera_name in self.model._camera_name2id:
                        camera_id = self.model.camera_name2id(camera_name)
                    self._get_viewer(mode).render(width, height, camera_id=camera_id)
                else:
                    # camera_id = self.model.name2id(camera_name, "camera")
                    camera_id = self.mjlib.mj_name2id(
                        self.model.ptr,
                        self._str2type("camera"),
                        self.util.to_binary_string(camera_name),
                    )
        if self._mujoco_framework.__name__ == "mujoco_py":
            if mode == "rgb_array":
                # window size used for old mujoco-py:
                data = self._get_viewer(mode).read_pixels(width, height, depth=False)
                # original image is upside-down, so flip it
                return data[::-1, :, :]
            elif mode == "depth_array":
                self._get_viewer(mode).render(width, height)
                # window size used for old mujoco-py:
                # Extract depth part of the read_pixels() tuple
                data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
                # original image is upside-down, so flip it
                return data[::-1, :]
            elif mode == "human":
                self._get_viewer(mode).render()
        else:
            if mode == "rgb_array":
                camera_id = camera_id or 0
                return self.sim.render(height, width, camera_id)
            elif mode == "depth_array":
                camera_id = camera_id or 0
                return self.sim.render(height, width, camera_id, depth=True)
            elif mode == "human":
                if self.viewer is None:
                    self.viewer = WindowViewer(self.sim, height, width)
                    self.viewer_setup()
                self.viewer.render_to_window()
            else:
                raise NotImplementedError

    def close(self):
        if self.viewer is not None:
            if self._mujoco_framework.__name__ == "mujoco_py":
                self._viewers = {}
            else:
                self.viewer.close()
            self.viewer = None

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                self.viewer = self._mujoco_framework.MjViewer(self.sim)
            elif mode == "rgb_array" or mode == "depth_array":
                self.viewer = self._mujoco_framework.MjRenderContextOffscreen(
                    self.sim, -1
                )

            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def get_body_com(self, body_name):
        if self._mujoco_framework.__name__ == "mujoco_py":
            return self.data.get_body_xpos(body_name)
        else:
            return self.sim.named.data.xpos[body_name]

    def state_vector(self):
        return np.concatenate([self.sim.data.qpos.flat, self.sim.data.qvel.flat])


class WindowViewer:
    """Renders DM Control Physics objects."""

    def __init__(self, sim, height=DEFAULT_WINDOW_HEIGHT, width=DEFAULT_WINDOW_WIDTH):
        self._window = None
        self._sim = sim
        self.height = height
        self.width = width
        self.set_free_camera_settings()

    def render_to_window(self):
        """Renders the Physics object to a window.
        The window continuously renders the Physics in a separate thread.
        This function is a no-op if the window was already created.
        """
        if not self._window:
            self._window = DMRenderWindow(self.width, self.height)
            self._window.load_model(self._sim)
            self._update_camera_properties(self._window.camera)

        self._window.run_frame()

    def refresh_window(self):
        """Refreshes the rendered window if one is present."""
        if self._window is None:
            return
        self._window.run_frame()

    def close(self):
        """Cleans up any resources being used by the renderer."""
        if self._window:
            self._window.close()
            self._window = None

    def set_free_camera_settings(
        self,
        trackbodyid=None,
        distance=None,
        azimuth=None,
        elevation=None,
        lookat=None,
        center=True,
    ):
        """Sets the free camera parameters.
        Args:
            distance: The distance of the camera from the target.
            azimuth: Horizontal angle of the camera, in degrees.
            elevation: Vertical angle of the camera, in degrees.
            lookat: The (x, y, z) position in world coordinates to target.
            center: If True and `lookat` is not given, targets the camera at the
                median position of the simulation geometry.
        """
        settings = {}
        if trackbodyid is not None:
            settings["trackbodyid"] = trackbodyid
        if distance is not None:
            settings["distance"] = distance
        if azimuth is not None:
            settings["azimuth"] = azimuth
        if elevation is not None:
            settings["elevation"] = elevation
        if lookat is not None:
            settings["lookat"] = np.array(lookat, dtype=np.float32)
        elif center:
            # Calculate the center of the simulation geometry.
            settings["lookat"] = np.array(
                [np.median(self._sim.data.geom_xpos[:, i]) for i in range(3)],
                dtype=np.float32,
            )

        self._camera_settings = settings

    def _update_camera_properties(self, camera):
        """Updates the given camera object with the current camera settings."""
        for key, value in self._camera_settings.items():
            if key == "lookat":
                getattr(camera, key)[:] = value
            else:
                setattr(camera, key, value)

    def __del__(self):
        """Automatically clean up when out of scope."""
        self.close()


class DMRenderWindow:
    """Class that encapsulates a graphical window."""

    def __init__(self, width, height, title=DEFAULT_WINDOW_TITLE):
        """Creates a graphical render window.
        Args:
            width: The width of the window.
            height: The height of the window.
            title: The title of the window.
        """
        import dm_control.viewer as dm_viewer
        import dm_control._render as dm_render

        self._dm_viewer = dm_viewer
        self._dm_render = dm_render
        self._viewport = self._dm_viewer.renderer.Viewport(width, height)
        self._window = self._dm_viewer.gui.RenderWindow(width, height, title)
        self._viewer = self._dm_viewer.viewer.Viewer(
            self._viewport, self._window.mouse, self._window.keyboard
        )
        self._draw_surface = None
        self._renderer = self._dm_viewer.renderer.NullRenderer()

    @property
    def camera(self):
        return self._viewer._camera._camera

    def close(self):
        self._viewer.deinitialize()
        self._renderer.release()
        self._draw_surface.free()
        self._window.close()

    def load_model(self, physics):
        """Loads the given Physics object to render."""
        self._viewer.deinitialize()

        self._draw_surface = self._dm_render.Renderer(
            max_width=_MAX_RENDERBUFFER_SIZE, max_height=_MAX_RENDERBUFFER_SIZE
        )
        self._renderer = self._dm_viewer.renderer.OffScreenRenderer(
            physics.model, self._draw_surface
        )

        self._viewer.initialize(physics, self._renderer, touchpad=False)

    def run_frame(self):
        """Renders one frame of the simulation."""
        glfw = self._dm_viewer.gui.glfw_gui.glfw
        glfw_window = self._window._context.window
        if glfw.window_should_close(glfw_window):
            sys.exit(0)

        self._viewport.set_size(*self._window.shape)
        self._viewer.render()
        pixels = self._renderer.pixels

        with self._window._context.make_current() as ctx:
            ctx.call(self._window._update_gui_on_render_thread, glfw_window, pixels)
        self._window._mouse.process_events()
        self._window._keyboard.process_events()
