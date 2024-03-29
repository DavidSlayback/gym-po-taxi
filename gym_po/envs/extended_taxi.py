__all__ = [
    "TaxiVecEnv",
    "ExtendedTaxiVecEnv",
    "HansenTaxiVecEnv",
    "ExtendedHansenTaxiVecEnv",
    "EXTENDED_TAXI_MAP",
    "TAXI_MAP",
]

from gymnasium.core import ObsType

"""Extended (8x8) Taxi from https://harshakokel.com/pdf/DRePReL-workshop.pdf"""
from functools import partial
from typing import Sequence, Callable, Tuple, Optional

import cv2
import gymnasium
import numpy as np
from numpy.typing import NDArray
from gymnasium.vector.utils import batch_space

from .grid_utils import DIRECTIONS_2D_NP
from .render_utils import CELL_PX, COLORS, tile_images

# Original taxi map, but not quite, need the pseudo-walls
TAXI_MAP = (
    "R: | : :G",
    " : | : : ",
    " : : : : ",
    " | : | : ",
    "Y| : |B: ",
)
FLOOR = COLORS.gray_mid_dark
WALL = COLORS.black
TAXI = COLORS.yellow
FULL_TAXI = COLORS.green
PASSENGER = COLORS.purple
TAXI_AND_PASSENGER = TAXI + PASSENGER
FAKE_WALL = COLORS.teal
LOC = COLORS.gray_light
DESTINATION = COLORS.blue
DIR = DIRECTIONS_2D_NP[:, :4]  # North, South, West, East

# Extended taxi map
EXTENDED_TAXI_MAP = (
    "R  |   G",
    "   |    ",
    "   |    ",
    "        ",
    "        ",
    "  |  |  ",
    "  |  |  ",
    "Y |  |B ",
)


def convert_str_map_to_walled_np_str(
    map: Sequence[str],
) -> Tuple[NDArray, NDArray, Callable]:
    """Return the full map (for image), the reduced map (for navigation), and a callable to convert coordinates from the 2nd to the first"""
    bordered_map = np.pad(
        np.asarray(map, dtype="c").astype(str), 1, constant_values="|"
    )
    if ":" in bordered_map:  # Handle pseudo-walls
        return (
            bordered_map,
            bordered_map[1:-1, 1:-1:2],
            lambda r, c: (r + 1, (2 * c) + 1),
        )
    return bordered_map, bordered_map[1:-1, 1:-1], lambda r, c: (r + 1, c + 1)


def compute_obs_space(tgrid: NDArray, n_locs: int = 4, hansen: bool = False) -> int:
    n_d_locs = n_locs  # destination can be in this many locations
    n_p_locs = n_locs + 1  # passenger can additionally be in taxi
    if hansen:
        n_t_obs = 2**4  # There can either be a wall or not in each of 4 directions
    else:
        y, x = tgrid.shape
        n_t_obs = y * x  # taxi can be in this many locations
    return int(n_t_obs * n_d_locs * n_p_locs)


def decode_state(
    states: NDArray[int], y: int = 5, n_locs: int = 4
) -> Sequence[NDArray[int]]:
    """Convert single int to taxi row, taxi col, passenger index, destination index"""
    d = states % n_locs
    tmp = states // n_locs
    p = tmp % (n_locs + 1)
    tmp = tmp // (n_locs + 1)
    c = tmp % y
    r = tmp // y
    return r.astype(int), c.astype(int), p.astype(int), d.astype(int)


def encode_state(r, c, p, d, y: int = 5, n_locs: int = 4):
    """Encode taxi row, taxi col, passenger index, and destination index as single int state"""
    return ((r * y + c) * (n_locs + 1) + p) * n_locs + d


def generate_hansen_map(bordered_map: NDArray, tgrid: NDArray, cc: Callable) -> NDArray:
    hansen_encodings = np.zeros(tgrid.shape, dtype=int)
    iswall = (bordered_map == "|").astype(int)
    for r in range(hansen_encodings.shape[0]):
        for c in range(hansen_encodings.shape[1]):
            br, bc = cc(r, c)
            hansen_encodings[r, c] = (
                (iswall[br - 1, bc])
                + 2 * iswall[br + 1, bc]
                + 2 * 2 * iswall[br, bc - 1]
                + 2 * 2 * 2 * iswall[br, bc + 1]
            )
    return hansen_encodings


def get_locations_from_np_str_map(map: NDArray) -> Sequence[NDArray]:
    return np.nonzero((map != "|") & (map != " ") & (map != ":"))


def str_map_to_img(
    map: NDArray, cell_pixel_size: int = CELL_PX, hansen_highlight: bool = False
) -> NDArray:
    """Convert bordered string map to actual rendered image"""
    n, y, x = map.shape[-3:]
    img = np.full((*map.shape, 3), 3, dtype=np.uint8)  # Fill value I don't use
    img[map == "|"] = WALL
    img[map == "P"] = PASSENGER
    img[map == "T"] = TAXI
    img[map == "TP"] = TAXI_AND_PASSENGER
    img[map == "F"] = FULL_TAXI
    img[map == "D"] = DESTINATION
    img[map == " "] = FLOOR
    img[map == ":"] = FAKE_WALL
    img[(img == 3).all(-1)] = LOC
    dir_idx = np.zeros((1, DIR.shape[-1]), dtype=int)
    if hansen_highlight:
        tloc = np.array(((map == "T") | (map == "F") | (map == "TP")).nonzero())
        hloc = (tloc[:, None] + np.concatenate((dir_idx, DIR))[..., None]).reshape(
            3, -1
        )
        img[tuple(hloc)] += 64
    img = tile_images(img)
    return cv2.resize(
        img, (y * cell_pixel_size, x * cell_pixel_size), interpolation=cv2.INTER_AREA
    )


class TaxiVecEnv(gymnasium.Env):
    """Vectorized Taxi environment"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5, "name": "Taxi"}

    ACTIONS_YX = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]], dtype=int)
    ACTION_NAMES = ["North", "South", "West", "East", "Pickup/Dropoff"]
    ACTION_DICT = {i: n for i, n in enumerate(ACTION_NAMES)}

    def __init__(
        self,
        num_envs: int = 1,
        time_limit: int = 200,  # How many steps
        num_passengers: int = 1,  # How many pickup/dropoffs?
        map: Sequence[str] = TAXI_MAP,  # Which map to use?
        hansen_obs: bool = False,  # If true, use hansen observations
        reward_goal: float = 1.0,  # Rewards
        reward_bad: float = -0.5,
        reward_any: float = -0.05,
        render_mode: Optional[str] = None,
    ):
        self.render_mode = render_mode
        self.is_vector_env = True
        self.num_envs = num_envs
        self.GOAL_MOVE, self.BAD_MOVE, self.ANY_MOVE = (
            reward_goal,
            reward_bad,
            reward_any,
        )
        self.desc, self.tgrid, self.cc = convert_str_map_to_walled_np_str(map)
        self.contains_pseudo_walls = (self.desc == ":").any()
        self.hansen_encodings = generate_hansen_map(self.desc, self.tgrid, self.cc)
        self.rows, self.cols = self.tgrid.shape
        self.locs = get_locations_from_np_str_map(self.tgrid)
        self.np_locs = np.array(self.locs).T
        self.nlocs = self.np_locs.shape[0]
        self.np_locs = np.concatenate((self.np_locs, [[-1, -1]]))  # Add extraneous loc

        # Timer
        self.time_limit = time_limit
        self.elapsed = np.zeros(self.num_envs, dtype=int)

        # Actions
        self.last_action = None
        self.single_action_space = gymnasium.spaces.Discrete(len(self.ACTIONS_YX))
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.na = self.single_action_space.n

        # Observations
        self.ns = compute_obs_space(self.tgrid, self.nlocs, False)  # Same # states

        self.no = compute_obs_space(self.tgrid, self.nlocs, hansen_obs)
        self.single_observation_space = gymnasium.spaces.Discrete(self.no)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.encode = partial(encode_state, y=self.cols, n_locs=self.nlocs)
        self.decode = partial(decode_state, y=self.cols, n_locs=self.nlocs)
        self.state_distribution = np.zeros(self.ns)
        valid_states = np.array(
            [
                self.encode(r, c, p, d)
                for r in range(self.rows)
                for c in range(self.cols)
                if self.tgrid[r, c] != "|"
                for p in range(self.nlocs)
                for d in range(self.nlocs)
                if d != p
            ]
        )
        self.state_distribution[valid_states] += 1
        self.state_distribution /= self.state_distribution.sum()
        self.hansen = False
        if hansen_obs:
            self.name = "HansenTaxi-v4"
            self.hansen = True

        # Internal state
        self.n_dropoffs = num_passengers
        self.s = np.zeros(self.num_envs, dtype=int)  # Maintain state as single int
        self.n_dropoffs_completed = np.zeros(
            self.num_envs
        )  # How many passengers have been delivered?
        self._viewer = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        """Fully reset all environments"""
        super().reset(seed=seed, options=options)
        self.lastaction = None
        self._reset_mask(np.ones(self.num_envs, bool))
        return self._obs(), {}

    def step(self, actions):
        self.elapsed += 1
        r, c, p, d = self.decode(self.s)
        # Take action, don't go out of bounds or into wall
        a = self.ACTIONS_YX[actions]
        rnew, cnew = np.clip(r + a[:, 0], 0, self.rows - 1), np.clip(
            c + a[:, 1], 0, self.cols - 1
        )
        cc = self.cc(rnew, cnew)  # Translate reduced coordinates to map coordinates
        not_wall = (
            self.desc[cc] != "|"
        )  # Handles extended map walls, all vertical movement
        crossed_wall = a[:, 1].astype(bool) & (
            self.desc[cc[0], cc[1] - a[:, 1]] == "|"
        )  # New coord is empty, but had to cross wall to get here
        not_wall &= ~crossed_wall
        r[not_wall], c[not_wall] = rnew[not_wall], cnew[not_wall]
        # Compute rewards
        tloc = np.column_stack((r, c))
        rew = np.full(self.num_envs, self.ANY_MOVE, dtype=np.float32)
        p_or_d = actions == 4  # Attempted pickup/dropoff
        # Goal is dropoff, with passenger in taxi, at destination location
        goal_move = p_or_d & (p == self.nlocs) & (self.np_locs[d] == tloc).all(-1)
        self.n_dropoffs_completed[goal_move] += 1
        # Can pickup if passenger in same location as taxi, not yet in taxi
        pickup_move = p_or_d & (p < self.nlocs) & (self.np_locs[p] == tloc).all(-1)
        p[pickup_move] = self.nlocs
        self.s = self.encode(r, c, p, d)
        # Any other attempt to pickup/dropoff is wrong. No change in state
        bad_move = p_or_d & ~goal_move & ~pickup_move
        rew[goal_move] = self.GOAL_MOVE
        rew[bad_move] = self.BAD_MOVE
        done = np.zeros(self.num_envs, bool)
        # Terminal if we did our # tasks or if we ran out of time
        done[self.n_dropoffs_completed == self.n_dropoffs] = True
        truncated = self.elapsed > self.time_limit
        self.lastaction = actions[0] if not done[0] else None
        # Reset passenger and destination (but not taxi) where not done but completed a task
        task_completed = goal_move & ~(done | truncated)
        self._reset_passenger_and_destination(
            task_completed, r[task_completed], c[task_completed]
        )
        self._reset_mask(done | truncated)
        return self._obs(), rew, done, truncated, {}

    def render(self, idx: Optional[Sequence[int]] = None):
        if idx is None:
            idx = np.arange(1)
        idx = np.array(idx)
        img = self.desc.copy()
        img = np.stack(tuple(img for _ in range(idx.size)))
        r, c, p, d = self.decode(self.s[idx])
        tc = self.cc(r, c)
        dc = (idx,) + self.cc(*self.np_locs[d].T)
        pc = self.cc(*self.np_locs[p].T)
        img[dc] = "D"
        p_in_taxi = p == self.nlocs
        img[(idx,) + tc] = "T"
        img[(~p_in_taxi).nonzero() + (pc[0][~p_in_taxi], pc[1][~p_in_taxi])] = "P"
        img[idx[p_in_taxi], tc[0][p_in_taxi], tc[1][p_in_taxi]] = "F"  # Full taxis
        same_coord = ~p_in_taxi & (pc[0] == tc[0]) & (pc[1] == tc[1])
        img[
            idx[same_coord], pc[0][same_coord], pc[1][same_coord]
        ] = "TP"  # Taxi in same spot
        text_space = 20
        img = str_map_to_img(img, hansen_highlight=self.hansen)
        img = np.concatenate(
            (img, np.zeros((img.shape[0], text_space, img.shape[2]), dtype=np.uint8)),
            axis=1,
        )
        text_anchor = (5, img.shape[1] - text_space)
        # Add last action
        text = (
            f"  ({self.ACTION_NAMES[self.lastaction]})\n"
            if self.lastaction is not None
            else ""
        )
        if text:
            cv2.putText(
                img,
                text,
                text_anchor,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.25,
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA,
            )
        if self.render_mode == "human":
            import pygame

            if self._viewer is None:
                pygame.init()
                self._viewer = pygame.display.set_mode(img.shape[:-1])
            sfc = pygame.surfarray.make_surface(img.swapaxes(0, 1))
            self._viewer.blit(sfc, (0, 0))
            pygame.display.update()
            return img
        return img

    def _reset_mask(self, mask: NDArray[bool]):
        """Fully reset some environments"""
        b = mask.sum()
        if b:
            self.s[mask] = self.np_random.multinomial(
                self.ns, self.state_distribution, b
            ).argmax(-1)
            self.elapsed[mask] = 0
            self.n_dropoffs_completed[mask] = 0

    def _reset_passenger_and_destination(self, mask: NDArray[bool], r=None, c=None):
        """Only reset passenger and destination locations for some environments"""
        b = mask.sum()
        if b:
            if (r is None) or (c is None):
                r, c = self.decode(self.s[mask])[:-2]
            p_idx = self.np_random.integers(self.nlocs, size=b)  # Place passenger
            d_idx = self.np_random.integers(self.nlocs, size=b)  # Place destination
            while (m := mask[mask] & (p_idx == d_idx)).any():
                d_idx[m] = self.np_random.integers(self.nlocs, size=m.sum())
            self.s[mask] = self.encode(r, c, p_idx, d_idx)  # Store in state

    def _obs(self):
        """Discrete observation for agent"""
        return self._hansen_obs(*self.decode(self.s)) if self.hansen else self.s

    def _hansen_obs(self, r, c, p, d):
        """Hansen-style observation (walls only)"""
        return (self.hansen_encodings[r, c] * (self.nlocs + 1) + p) * self.nlocs + d


HansenTaxiVecEnv = partial(TaxiVecEnv, hansen_obs=True)
ExtendedTaxiVecEnv = partial(TaxiVecEnv, map=EXTENDED_TAXI_MAP)
ExtendedHansenTaxiVecEnv = partial(HansenTaxiVecEnv, map=EXTENDED_TAXI_MAP)


if __name__ == "__main__":
    e = TaxiVecEnv(256, map=EXTENDED_TAXI_MAP, hansen_obs=True)
    o = e.reset()
    for t in range(100000):
        o, r, d, info = e.step(e.action_space.sample())
        img = e.render("rgb_array")
    # e.render()
    # img = e.render('rgb_array')
    # from PIL import Image
    # tim = Image.fromarray(img)
    # tim.save('test.png')
    print(3)
