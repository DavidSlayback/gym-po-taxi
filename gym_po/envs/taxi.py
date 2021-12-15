__all__ = ['TaxiEnv', 'TaxiVecEnv', 'HansenTaxiVecEnv']

import sys
from contextlib import closing
from io import StringIO
from gym import utils, Env
from gym.utils.seeding import np_random
from gym.spaces import Discrete
from gym.vector.utils import batch_space
from gym.envs.toy_text import discrete
import numpy as np

# Base map for Taxi
MAP = [
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]

class TaxiEnv(discrete.DiscreteEnv):
    """
    The Taxi Problem
    from "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
    by Tom Dietterich
    Description:
    There are four designated locations in the grid world indicated by R(ed), G(reen), Y(ellow), and B(lue). When the episode starts, the taxi starts off at a random square and the passenger is at a random location. The taxi drives to the passenger's location, picks up the passenger, drives to the passenger's destination (another one of the four specified locations), and then drops off the passenger. Once the passenger is dropped off, the episode ends.
    Observations:
    There are 500 discrete states since there are 25 taxi positions, 5 possible locations of the passenger (including the case when the passenger is in the taxi), and 4 destination locations.
    Note that there are 400 states that can actually be reached during an episode. The missing states correspond to situations in which the passenger is at the same location as their destination, as this typically signals the end of an episode.
    Four additional states can be observed right after a successful episodes, when both the passenger and the taxi are at the destination.
    This gives a total of 404 reachable discrete states.
    Passenger locations:
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)
    - 4: in taxi
    Destinations:
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)
    Actions:
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: pickup passenger
    - 5: drop off passenger
    Rewards:
    There is a default per-step reward of -1,
    except for delivering the passenger, which is +20,
    or executing "pickup" and "drop-off" actions illegally, which is -10.
    Rendering:
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - other letters (R, G, Y and B): locations for passengers and destinations
    state space is represented by:
        (taxi_row, taxi_col, passenger_location, destination)
    """

    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self):
        self.desc = np.asarray(MAP, dtype="c")

        self.locs = locs = [(0, 0), (0, 4), (4, 0), (4, 3)]

        num_states = 500
        num_rows = 5
        num_columns = 5
        max_row = num_rows - 1
        max_col = num_columns - 1
        initial_state_distrib = np.zeros(num_states)
        num_actions = 6
        P = {
            state: {action: [] for action in range(num_actions)}
            for state in range(num_states)
        }
        for row in range(num_rows):
            for col in range(num_columns):
                for pass_idx in range(len(locs) + 1):  # +1 for being inside taxi
                    for dest_idx in range(len(locs)):
                        state = self.encode(row, col, pass_idx, dest_idx)
                        if pass_idx < 4 and pass_idx != dest_idx:
                            initial_state_distrib[state] += 1
                        for action in range(num_actions):
                            # defaults
                            new_row, new_col, new_pass_idx = row, col, pass_idx
                            reward = (
                                -1
                            )  # default reward when there is no pickup/dropoff
                            done = False
                            taxi_loc = (row, col)

                            if action == 0:
                                new_row = min(row + 1, max_row)
                            elif action == 1:
                                new_row = max(row - 1, 0)
                            if action == 2 and self.desc[1 + row, 2 * col + 2] == b":":
                                new_col = min(col + 1, max_col)
                            elif action == 3 and self.desc[1 + row, 2 * col] == b":":
                                new_col = max(col - 1, 0)
                            elif action == 4:  # pickup
                                if pass_idx < 4 and taxi_loc == locs[pass_idx]:
                                    new_pass_idx = 4
                                else:  # passenger not at location
                                    reward = -10
                            elif action == 5:  # dropoff
                                if (taxi_loc == locs[dest_idx]) and pass_idx == 4:
                                    new_pass_idx = dest_idx
                                    done = True
                                    reward = 20
                                elif (taxi_loc in locs) and pass_idx == 4:
                                    new_pass_idx = locs.index(taxi_loc)
                                else:  # dropoff at wrong location
                                    reward = -10
                            new_state = self.encode(
                                new_row, new_col, new_pass_idx, dest_idx
                            )
                            P[state][action].append((1.0, new_state, reward, done))
        initial_state_distrib /= initial_state_distrib.sum()
        discrete.DiscreteEnv.__init__(
            self, num_states, num_actions, P, initial_state_distrib
        )

    def encode(self, taxi_row, taxi_col, pass_loc, dest_idx):
        # (5) 5, 5, 4
        i = taxi_row
        i *= 5  # times y
        i += taxi_col
        i *= 5  # times number dest + 1
        i += pass_loc
        i *= 4  # times number dest
        i += dest_idx
        return i

    def decode(self, i):
        out = []
        out.append(i % 4)
        i = i // 4
        out.append(i % 5)
        i = i // 5
        out.append(i % 5)
        i = i // 5
        out.append(i)
        assert 0 <= i < 5
        return reversed(out)

    def render(self, mode="human"):
        """Render this environment as text"""
        outfile = StringIO() if mode == "ansi" else sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode("utf-8") for c in line] for line in out]
        taxi_row, taxi_col, pass_idx, dest_idx = self.decode(self.s)

        def ul(x):
            return "_" if x == " " else x

        if pass_idx < 4:
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                out[1 + taxi_row][2 * taxi_col + 1], "yellow", highlight=True
            )
            pi, pj = self.locs[pass_idx]
            out[1 + pi][2 * pj + 1] = utils.colorize(
                out[1 + pi][2 * pj + 1], "blue", bold=True
            )
        else:  # passenger in taxi
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                ul(out[1 + taxi_row][2 * taxi_col + 1]), "green", highlight=True
            )

        di, dj = self.locs[dest_idx]
        out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], "magenta")
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        if self.lastaction is not None:
            outfile.write(
                f"  ({['South', 'North', 'East', 'West', 'Pickup', 'Dropoff'][self.lastaction]})\n"
            )
        else:
            outfile.write("\n")

        # No need to return anything for human
        if mode != "human":
            with closing(outfile):
                return outfile.getvalue()


ns, x, y, locs, na = 500, 5, 5, 4, 5
ACTIONS = np.array([
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (0, 0)
])

# actions expressed as flat indices
FLAT_ACTIONS = np.ravel_multi_index((ACTIONS+1).T, (x,y)) - np.ravel_multi_index((1,1), (x,y))


def encode(r, c, p, d):
    """row, col, pass_idx, dest_idx -> discrete"""
    return ((r * y + c) * (locs+1) + p) * locs + d


def decode(i):
    """Discrete state -> row, col, pass_idx, dest_idx"""
    d = i % locs
    tmp = i // locs
    p = tmp % 5
    tmp = tmp // 5
    c = tmp % 5
    r = tmp // 5
    return r, c, p, d

# Initial state distribution
STATE_DISTRIBUTION = np.zeros(ns)
VALID_STATES = np.array([encode(r, c, p, d) for r in range(y) for c in range(x) for p in range(locs) for d in range(locs) if d != p])
STATE_DISTRIBUTION[VALID_STATES] += 1
STATE_DISTRIBUTION /= STATE_DISTRIBUTION.sum()

class TaxiVecEnv(Env):
    """Vectorized original taxi environment"""
    metadata = {"render.modes": ["human", "ansi"]}

    desc = np.asarray(MAP, dtype='c')
    locs = np.array([(0, 0), (0, 4), (4, 0), (4, 3), (-1, -1)])  # R, G, Y, B, invalid
    # Scale rewards down by factor of 10
    BAD_MOVE = -0.5
    GOAL_MOVE = 1
    ANY_MOVE = -0.05

    def __init__(self, num_envs: int, time_limit: int = 0):
        # Preliminaries
        self.is_vector_env = True
        self.num_envs = num_envs
        self.time_limit = time_limit or int(1e6)
        self.lastaction = None  # For rendering
        # down, up, right, left, pickup/dropoff (simplify action space)
        self.single_action_space = Discrete(na)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.single_observation_space = Discrete(ns)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        # Agent is at (x,y). Passenger is in 1 of 4 locations or taxi. Destination is in 1 of 4 locations
        self.seed()
        self.s = np.zeros(num_envs, int)
        self.elapsed = np.zeros(num_envs, int)

    def seed(self, seed=None):
        self.rng, seed = np_random(seed)
        return seed

    def reset(self):
        self.lastaction = None
        self._reset_mask(np.ones(self.num_envs, bool))
        return self._obs()

    def render(self, mode="human"):
        """Render first environment in set"""
        outfile = StringIO() if mode == "ansi" else sys.stdout
        out = self.desc.copy().tolist()
        out = [[c.decode("utf-8") for c in line] for line in out]
        taxi_row, taxi_col, pass_idx, dest_idx = decode(self.s[0])

        def ul(x):
            return "_" if x == " " else x

        if pass_idx < 4:
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                out[1 + taxi_row][2 * taxi_col + 1], "yellow", highlight=True
            )
            pi, pj = self.locs[pass_idx]
            out[1 + pi][2 * pj + 1] = utils.colorize(
                out[1 + pi][2 * pj + 1], "blue", bold=True
            )
        else:  # passenger in taxi
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                ul(out[1 + taxi_row][2 * taxi_col + 1]), "green", highlight=True
            )

        di, dj = self.locs[dest_idx]
        out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], "magenta")
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        if self.lastaction is not None:
            outfile.write(
                f"  ({['South', 'North', 'East', 'West', 'Pickup/Dropoff'][self.lastaction]})\n"
            )
        else:
            outfile.write("\n")

        # No need to return anything for human
        if mode != "human":
            with closing(outfile):
                return outfile.getvalue()

    def _reset_mask(self, mask: np.ndarray):
        b = mask.sum()
        if b:
            self.s[mask] = self.rng.multinomial(ns, STATE_DISTRIBUTION, b).argmax(-1)
            self.elapsed[mask] = 0

    def step(self, actions):
        self.elapsed += 1
        r, c, p, d = decode(self.s)
        r, c = np.clip(r + ACTIONS[actions][:, 0], 0, x-1), np.clip(c + ACTIONS[actions][:, 1], 0, y-1)
        tloc = np.column_stack((r, c))
        rew = np.full(self.num_envs, self.ANY_MOVE, dtype=np.float32)
        p_or_d = actions == 4  # Attempted pickup/dropoff
        # Goal is dropoff, with passenger in taxi, at destination location
        goal_move = p_or_d & (p == locs) & (self.locs[d] == tloc).all(-1)
        # Can pickup if passenger in same location as taxi
        pickup_move = p_or_d & (p < locs) & (self.locs[p] == tloc).all(-1)
        p[pickup_move] = locs
        self.s = encode(r, c, p, d)  # Update state
        # Any other attempt to pickup/dropoff is wrong. No change in state
        bad_move = p_or_d & ~goal_move & ~pickup_move
        rew[goal_move] = self.GOAL_MOVE
        rew[bad_move] = self.BAD_MOVE
        done = np.zeros(self.num_envs, bool)
        done[goal_move] = True
        done[self.elapsed >= self.time_limit] = True
        self._reset_mask(done)
        self.lastaction = actions[0] if not done[0] else None  # Last action or None if reset
        return self._obs(), rew, done, [{}] * self.num_envs

    def _obs(self):
        return self.s

# +1 for left, +2 for below, +4 for right, +8 for above
WALL_CODES = np.array([
    [9, 12, 9, 8, 12],
    [1, 4, 1, 0, 4],
    [1, 0, 0, 0, 4],
    [5, 1, 4, 1, 4],
    [7, 3, 6, 3, 6]
], dtype=int)
possible_obs = 16  # len(np.unique(WALL_CODES))
no = possible_obs * locs * (locs+1)  # 2^4 * 4 * 5
def encode_obs(o, p, d):
    """Encode observation using wall code"""
    return (o * (locs + 1) + p) * locs + d


class HansenTaxiVecEnv(TaxiVecEnv):
    """Make Taxi partially observable as in
    "Synthesis of Hierarchical Finite-State Controllers for POMDPs"
    Hansen & Zhou

    Assume Taxi cannot observe its location
    Instead, it can only observe wall placement in all 4 directions immediately adjacent
    Passenger and goal locations remain visible
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.single_observation_space = Discrete(no)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)

    def _obs(self):
        r, c, p, d = decode(self.s)
        o = WALL_CODES[r, c]
        return encode_obs(o, p, d)

