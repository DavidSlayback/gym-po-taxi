import time

import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
from gym.vector.utils import batch_space
from gym.utils.seeding import np_random
from typing import Optional, Tuple, List, Union, Sequence

WALL = 0
EMPTY = 1
STAIR = 2
GOAL = 3
AGENT = 4

# RGB Colors for these
FRCOLORS = np.array([
    [0, 0, 0],  # Black walls
    [128,128,128],  # Gray empty
    [0, 0, 128],  # Blue stairs
    [0, 128, 0],  # Green goal
    [128, 0, 0]  # Red agent
], dtype=np.uint8)
VIEW_COLOR_OFFSET = 60  # Add to squares in view

def tile_images(img_nhwc: Sequence[np.ndarray]) -> np.ndarray:  # pragma: no cover
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    :param img_nhwc: list or array of images, ndim=4 once turned into array. img nhwc
        n = batch index, h = height, w = width, c = channel
    :return: img_HWc, ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    n_images, height, width, n_channels = img_nhwc.shape
    # img_nhwc[:, 0:2, 0:2, :] = np.array([255,255,255])
    # img_nhwc[:, -3:, -3:, :] = np.array([255,255,255])
    # new_height was named H before
    new_height = int(np.ceil(np.sqrt(n_images)))
    # new_width was named W before
    new_width = int(np.ceil(float(n_images) / new_height))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0] * 0 for _ in range(n_images, new_height * new_width)])
    # img_HWhwc
    out_image = img_nhwc.reshape((new_height, new_width, height, width, n_channels))
    # img_HhWwc
    out_image = out_image.transpose(0, 2, 1, 3, 4)
    # img_Hh_Ww_c
    out_image = out_image.reshape((new_height * height, new_width * width, n_channels))
    return out_image

def grid_to_rgb(grid: np.ndarray, n_pixel: int = 8, grid_highlight: Optional[np.ndarray] = None) -> np.ndarray:
    """Convert base grid observation to pixels"""
    grid = grid.swapaxes(-2, -1)
    if grid_highlight is not None: grid_highlight = grid_highlight.swapaxes(-2, -1)
    (n, h, w), c = grid.shape, 3
    img = np.zeros((n, h*n_pixel, w*n_pixel, c), dtype=np.uint8)
    if grid_highlight is None:
        for b in range(n):
            for y in range(h):
                for x in range(w):
                    img[b, y*n_pixel:y*n_pixel+n_pixel, x*n_pixel:x*n_pixel+n_pixel] = FRCOLORS[grid[b,y,x]] # Fill with color
    else:
        for b in range(n):
            for y in range(h):
                for x in range(w):
                    img[b, y * n_pixel:y * n_pixel + n_pixel, x * n_pixel:x * n_pixel + n_pixel] = FRCOLORS[
                        grid[b, y, x]] + VIEW_COLOR_OFFSET * grid_highlight[b, y, x]
    return tile_images(img)

# Basic 13x13 FourRooms grid. 0 is wall. 1 is empty
MAP = np.array([
    [WALL]*13,
    [WALL]+([EMPTY]*5) + [WALL] + ([EMPTY]*5) + [WALL],
    [WALL]+([EMPTY]*5) + [WALL] + ([EMPTY]*5) + [WALL],
    [WALL]+([EMPTY]*11) + [WALL],
    [WALL]+([EMPTY]*5) + [WALL] + ([EMPTY]*5) + [WALL],
    [WALL]+([EMPTY]*5) + [WALL] + ([EMPTY]*5) + [WALL],
    [WALL, WALL, EMPTY] + ([WALL]*4) + ([EMPTY]*5) + [WALL],
    [WALL] + ([EMPTY]*5) + ([WALL]*3) + [EMPTY] + ([WALL]*3),
    [WALL]+([EMPTY]*5) + [WALL] + ([EMPTY]*5) + [WALL],
    [WALL]+([EMPTY]*5) + [WALL] + ([EMPTY]*5) + [WALL],
    [WALL]+([EMPTY]*11) + [WALL],
    [WALL]+([EMPTY]*5) + [WALL] + ([EMPTY]*5) + [WALL],
    [WALL]*13,
], dtype=int)
y, x = MAP.shape
NE = (1, 11)  # NE stairs are upstairs
SW = (11, 1)  # SW stairs are downstairs
"""
Ndarrays are contiguous (single-segment, memory-layout), access with strides
Defaults to row-major order

Numpy arrays should be in order of access frequency.
But screw that for now!
"""

# Up, right, down, left, upstairs, downstairs (z, y, x)
ACTIONS = np.array([
    [0, -1, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 0, -1],
    [1, 0, 0],
    [-1, 0, 0]
], dtype=int)
UPSTAIR_OFFSET = int((10*13) - 10)  # NE-> SW
DOWNSTAIR_OFFSET = int((-10*13) + 10)  # SW -> NE

# Add stairs to bottom, mid, and top floors
B_MAP, M_MAP, T_MAP = MAP.copy(), MAP.copy(), MAP.copy()
B_MAP[NE] = STAIR; M_MAP[NE] = M_MAP[SW] = STAIR; T_MAP[SW] = STAIR

def generate_layout(grid_z: int = 1):
    """Generate multi-story layout (z, y, x)"""
    if grid_z == 1:
        return MAP.copy()[None, ...]
    elif grid_z == 2:
        return np.stack((B_MAP, T_MAP), axis=0)
    else:
        return np.stack((B_MAP,*(M_MAP for _ in range(grid_z-2)), T_MAP), axis=0)


def generate_flat_actions(grid_z: int = 1):
    """Generate flattened versions of actions"""
    a = np.zeros(6, dtype=int)
    if grid_z > 1:
        p_a = np.ravel_multi_index(ACTIONS[[1,2,4]].T, (grid_z, y, x))
        a[[1,2,4]] = p_a
        a[[3, 0, 5]] = -p_a
    else:
        p_a = np.ravel_multi_index(ACTIONS[[1,2]].T, (grid_z, y, x))
        a[[1,2]] = p_a
        a[[3, 0]] = -p_a
    # a = np.ravel_multi_index((ACTIONS+1).T, (grid_z, y, x)) - np.ravel_multi_index((1, 1, 1), (grid_z, y, x))
    a[-2] += UPSTAIR_OFFSET
    a[-1] += DOWNSTAIR_OFFSET
    return a


def action_probability_matrix(action_failure_probability: float = (1./3), action_n: int = 4):
    probs = np.full((action_n, action_n), action_failure_probability / (action_n - 1), dtype=np.float64)
    np.fill_diagonal(probs, 1 - action_failure_probability)
    return probs

def vectorized_multinomial(selected_prob_matrix: np.ndarray, random_numbers: np.ndarray):
    """Vectorized sample from [B,N] probabilitity matrix
    Lightly edited from https://stackoverflow.com/a/34190035/2504700
    Args:
        selected_prob_matrix: (Batch, p) size probability matrix (i.e. T[s,a] or O[s,a,s']
        random_numbers: (Batch,) size random numbers from np.random.rand()
    Returns:
        (Batch,) size sampled integers
    """
    s = selected_prob_matrix.cumsum(axis=1)  # Sum over p dim for accumulated probability
    return (s < np.expand_dims(random_numbers, axis=-1)).sum(axis=1)  # Returns first index where random number < accumulated probability


class MultistoryFourRoomsVecEnv(Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 15}
    """Discrete multistory fourrooms environment"""
    def __init__(self, num_envs: int, grid_z: int = 1, time_limit: int = 10000, seed=None,
                 fixed_goal: Union[List[int], Tuple[int, int], Tuple[int, int, int], int] = 0, action_failure_probability: float = (1./3),
                 wall_reward: float = 0., goal_z=None, agent_z=None):
        # Preliminaries
        self.num_envs = num_envs
        self.is_vector_env = True
        self.time_limit = time_limit
        self.wall_reward = wall_reward
        agent_z = agent_z or 0
        goal_z = goal_z or (grid_z-1)
        self.gz, self.az = goal_z, agent_z

        # Actions
        self.single_action_space = Discrete(4)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.aprob_with_failure = action_probability_matrix(action_failure_probability, 4)
        # Grid
        self.grid = generate_layout(grid_z)
        empty_locs = np.array((self.grid == EMPTY).nonzero())  # Where can things spawn
        valid_locs = np.array((self.grid != WALL).nonzero())  # Where can agent be
        self.grid_flat = self.grid.ravel()
        self.actions = generate_flat_actions(grid_z)
        # Convert to and from scalar indices
        self.encode = lambda zyx_coord: np.ravel_multi_index(zyx_coord, (grid_z, y, x))
        self.decode = lambda i: np.unravel_index(i, (grid_z, y, x))
        self.flat_empty_locs = self.encode(empty_locs)
        self.flat_valid_locs = self.encode(valid_locs)
        self.flat_goal_locs = self.encode(empty_locs[:, np.isin(empty_locs[0], goal_z, True)])
        self.flat_agent_locs = self.encode(empty_locs[:, np.isin(empty_locs[0], agent_z, True)])
        # Stairs locations
        self.upstairs = self.encode(np.column_stack([(z, *NE) for z in np.arange(grid_z-1)])) if grid_z > 1 else np.empty(())
        self.downstairs = self.encode(np.column_stack([(z, *SW) for z in np.arange(1, grid_z)])) if grid_z > 1 else np.empty(())

        # Observations
        self._ns = valid_locs.shape[-1]
        # Convert raw flattened grid location to observation
        self.fcoord_to_ecoord = np.isin(np.arange(self.grid.size), self.flat_valid_locs, True).cumsum() - 1
        self.single_observation_space = Discrete(self._ns)
        self.observation_space = batch_space(self.single_observation_space, num_envs)

        self.seed(seed)
        # States
        self.agent = np.zeros(self.num_envs, int)
        self.elapsed = np.zeros(self.num_envs, int)
        if fixed_goal:
            fixed_goal = self._convert_goal_tuple(fixed_goal)
            self.goal_fn = lambda b: np.full(b, fixed_goal, dtype=int)
        else:
            self.goal_fn = lambda b: self.rng.choice(self.flat_goal_locs, b)
        self.goal = np.full(self.num_envs, fixed_goal, dtype=int)
        self._viewer = None

    def seed(self, seed=None):
        self.rng, seed = np_random(seed)
        return seed

    def render(self, mode='human', idx: np.ndarray = np.array([0], dtype=np.int64),
               cell_pixel_size: int = 16):
        idx = np.atleast_1d(idx)
        floors, r, c = self.decode(self.agent[idx])
        g_floors, g_r, g_c = self.decode(self.goal[idx])
        show_goal = np.flatnonzero(floors == g_floors)
        g = self.grid[floors]
        g[show_goal, g_r[show_goal], g_c[show_goal]] = GOAL
        g[idx, r, c] = AGENT
        img = grid_to_rgb(g)
        if mode == 'rgb' or mode == 'rgb_array': return img
        else:
            import pygame
            if self._viewer is None:
                pygame.init()
                self._viewer = pygame.display.set_mode(img.shape[:-1])
            sfc = pygame.surfarray.make_surface(img)
            self._viewer.blit(sfc, (0,0))
            pygame.display.update()
            return img

    def _convert_goal_tuple(self, goal: Union[List[int], Tuple[int, int], Tuple[int, int, int], int]):
        if isinstance(goal, (Tuple, List)):
            # Convert tuple goals to single indices. If no z is provided, default to top floor
            if len(goal) == 2: goal = list((self.grid.shape[0]-1, *goal))
            goal = self.encode(goal)
        assert goal in self.flat_empty_locs  # Ensure it's a valid place for a goal
        return goal

    def set_goal(self, goal):
        self.goal_fn = lambda b: np.full(b, goal, dtype=int)
        self.goal[:] = self.goal_fn(self.num_envs)

    def sample_goal(self, grid_z: None):
        """Sample random goal for all envs. Override original goal_z

        If no grid_z is provided, can sample on all floors. If it is, can only sample on that/those floors
        """
        if grid_z is None:
            g = self.rng.choice(self.flat_empty_locs, self.num_envs)
        else:
            EMPTY_LOCS = self.empty_locs[:, np.isin(self.empty_locs[0], grid_z, True)]
            FLAT_EMPTY = np.ravel_multi_index(EMPTY_LOCS, (grid_z, y, x))
            g = self.rng.choice(FLAT_EMPTY, self.num_envs)
        return g

    def reset(self):
        self._reset_mask(np.ones(self.num_envs, bool))
        return self._obs()

    def _reset_mask(self, mask: np.ndarray):
        b = mask.sum()
        if b:
            self.goal[mask] = self.goal_fn(b)
            # Make sure agent doesn't start on goal
            self.agent[mask] = self.rng.choice(self.flat_agent_locs, b)
            while (m := mask & (self.agent == self.goal)).any():
                self.agent[m] = self.rng.choice(self.flat_agent_locs, m.sum())
            self.elapsed[mask] = 0

    def _obs(self):
        return self.fcoord_to_ecoord[self.agent]

    def step(self, actions: np.ndarray):
        self.elapsed += 1
        rnd = self.rng.random(self.num_envs)
        a = vectorized_multinomial(self.aprob_with_failure[np.array(actions)], rnd)
        new_loc = self.agent + self.actions[a]  # Move
        moved = self._check_bounds(new_loc)  # Ensure bounds
        new_loc[~moved] = self.agent[~moved]  # Revert moves that are out of bounds
        upstairs = moved & np.isin(new_loc, self.upstairs)
        downstairs = moved & np.isin(new_loc, self.downstairs)
        new_loc[upstairs] += self.actions[-2]
        new_loc[downstairs] += self.actions[-1]
        self.agent[:] = new_loc
        r = np.zeros(self.num_envs, dtype=np.float32)
        d = (self.agent == self.goal)  # Done where we reached goal
        r[d] = 1.
        r[~moved] = self.wall_reward  # Potentially penalize hitting walls
        d |= self.elapsed >= self.time_limit  # Also done where we're out of time
        self._reset_mask(d)
        return self._obs(), r, d, [{}] * self.num_envs

    def _check_bounds(self, proposed_locations: np.ndarray):
        valid = (0 <= proposed_locations) & (proposed_locations < self.grid_flat.size)  # In bounds
        valid[valid] = self.grid_flat[proposed_locations[valid]] != WALL
        return valid

multipliers = np.array([1, 4, 16, 64])
class HansenMultistoryFourRoomsVecEnv(MultistoryFourRoomsVecEnv):
    """Use Hansen taxi-style observations (only immediately adjacent walls, goals, and stairs)"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ns = (multipliers * 3).sum()+1
        self.single_observation_space = Discrete(self._ns)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)

    def hansen_encode(self):
        adj = self.agent[..., None] + self.actions[:-2]
        g = self.grid_flat[adj]  # Comes with stairs, empty, walls
        g[self.goal[..., None] == adj] = GOAL  # Fill in goals
        return (g * multipliers).sum(-1)

    def _obs(self):
        return self.hansen_encode()


class GridMultistoryFourRoomsVecEnv(MultistoryFourRoomsVecEnv):
    """Use nxn observations around agent"""
    def __init__(self, obs_n: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (obs_n % 2) == 1
        self.o_shape = (obs_n, obs_n)
        o_min = 0
        o_max = GOAL
        self.single_observation_space = Box(o_min, o_max, self.o_shape, dtype=int)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        # Agent view offset
        self._offset = obs_n // 2
        self._f_offset = self.grid.shape[-2] * self.grid.shape[-1]
        v = np.mgrid[:1, :obs_n, :obs_n]
        self.view = (self.encode(v) - self.encode([0, self._offset, self._offset]))[0]

    def _obs(self):
        floors, _, _ = self.decode(self.agent)
        floors = floors[..., None, None]
        f_min = floors * self._f_offset
        f_max = np.minimum(f_min + self._f_offset, self.grid.size)
        av = self.view + self.agent[..., None, None]  # Add offset to get agent view indices
        # av[(av < 0) | (av >= self.grid.size)] = 0  # Out-of-bounds are walls
        av[(av < f_min) | (av >= f_max)] = 0  # Out-of-floor are walls (retrieved as grid_idx 0)
        goal_idx = self.goal[..., None, None] == av  # Fill in the goal
        # av[()]
        o = self.grid_flat[av]  # Walls everywhere we won't see. Note that our agent CAN see through walls
        o[goal_idx] = GOAL
        return o

    def render(self, mode='human', idx: np.ndarray = np.array([0], dtype=np.int64),
               cell_pixel_size: int = 16):
        idx = np.atleast_1d(idx)
        floors, r, c = self.decode(self.agent[idx])
        g_floors, g_r, g_c = self.decode(self.goal[idx])
        show_goal = np.flatnonzero(floors == g_floors)
        g = self.grid[floors]
        g[show_goal, g_r[show_goal], g_c[show_goal]] = GOAL
        g[idx, r, c] = AGENT
        img = grid_to_rgb(g)
        if mode == 'rgb' or mode == 'rgb_array': return img
        else:
            import pygame
            if self._viewer is None:
                pygame.init()
                self._viewer = pygame.display.set_mode(img.shape[:-1])
            sfc = pygame.surfarray.make_surface(img)
            self._viewer.blit(sfc, (0,0))
            pygame.display.update()
            return img




if __name__ == "__main__":
    a = generate_flat_actions(1)
    e = MultistoryFourRoomsVecEnv(8, 2, fixed_goal=(7,9))
    o = e.reset()
    o, r, d, info = e.step(e.action_space.sample())
    for t in range(10000):
        o, r, d, info = e.step(e.action_space.sample())
        if d.any(): print(r[d])
        e.render(idx=np.arange(8))
        time.sleep(0.05)
        # print(o)
    print(3)
