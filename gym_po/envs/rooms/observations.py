__all__ = [
    "get_hansen_obs",
    "get_number_discrete_states_and_conversion",
    "get_number_abstract_states",
    "get_grid_obs",
    "get_hansen_vector_obs",
]
from typing import Tuple, Optional

import numpy as np
from numpy.typing import NDArray

from .action_utils import ACTIONS_CARDINAL, ACTIONS_ORDINAL


def get_number_discrete_states_and_conversion(
    grid: NDArray[int],
) -> Tuple[int, NDArray[int]]:
    """Count the number of possible state observation given a gridified version of a layout

    Args:
        grid: Gridified layout
    Returns:
        n_state: Number of discrete states
        state_grid: Grid that converts (y,x) coords to discrete state
    """
    n_states = (grid >= 0).sum()
    state_grid = ((grid >= 0).cumsum() - 1).reshape(grid.shape)
    return n_states, state_grid


def get_number_abstract_states(grid: NDArray[int]) -> int:
    """Count number of rooms

    Args:
        grid: Gridified layout
    Returns:
        n_state: Number of abstract "room" states
    """
    n_states = len(np.unique(grid)) - 1  # Ignore walls
    return n_states


def get_hansen_obs(
    agent_yx: NDArray[int], grid: NDArray[int], goal_yx: NDArray[int], hansen_n: int = 8
) -> int:
    """Get hansen observation of agent(s) (empty, wall), goal in (null, N, E, S, W) based on grid

    Args:
        agent_yx: (y, x) coordinate of agent(s) [B, 2]
        grid: (y, x) numpy grid
        goal_yx: (y, x) goal location(s)
        hansen_n: 8 or 4
    Returns:
        obs
    """
    a = ACTIONS_CARDINAL if hansen_n == 4 else ACTIONS_ORDINAL
    a = a[None, :]
    coords = agent_yx[:, None] + a
    # is_goal = (goal_yx[:, None] == coords).all(-1)
    where_is_goal = np.nonzero((goal_yx[:, None] == coords).all(-1))
    goal_mult = np.ones(goal_yx.shape[0])
    goal_mult[where_is_goal[0]] = where_is_goal[1] + 1
    squares = grid[tuple(coords.transpose(2, 0, 1))]
    squares += 1
    squares[squares > 0] = 1  # Empty squares
    # squares[is_goal] = 2  # Add goal
    multipliers = np.array(
        [2**i for i in range(a.shape[1])]
    )  # There's only one goal, let's multiply it separately after
    return squares.dot(multipliers) * goal_mult


def get_grid_obs(
    agent_yx: NDArray[int], grid: NDArray[int], goal_yx: NDArray[int], n: int = 3
) -> NDArray[int]:
    """Return grid observation

    Args:
        agent_yx: (y, x) coordinate of agent(s) [B, 2]
        grid: (y, x) numpy grid
        goal_yx: (y, x) goal location(s)
        n: 8 or 4
    Returns:
        obs (0 for wall, 1 for empty, 2 for goal)
    """
    offset = n // 2  # Center on agent
    mg = np.mgrid[:n, :n] - offset  # 2xNxN
    coords = (agent_yx[..., None, None] + mg[None, ...]).swapaxes(0, 1)
    # All invalid coords should point to a wall (i.e., (0,0))
    # coords[coords < 0] = 0
    coords[
        :,
        (coords[0] < 0)
        | (coords[1] < 0)
        | (coords[0] >= grid.shape[0])
        | (coords[1] >= grid.shape[1]),
    ] = 0
    is_goal = (goal_yx.swapaxes(0, 1)[..., None, None] == coords).all(0)
    squares = grid[tuple(coords)] + 1
    squares[squares > 0] = 1
    squares[is_goal] = 2
    return squares


def get_hansen_vector_obs(
    agent_yx: NDArray[int],
    grid: NDArray[int],
    goal_yx: Optional[np.ndarray] = None,
    hansen_n: int = 8,
) -> NDArray[int]:
    """Same as above, but a vector representation (like the grid obs, but flattened)

    Args:
        agent_yx: (y,x) coordinate of agent(s) [B,2]
        grid: (y,x) numpy grid
        goal_yx (y,x) goal location(s)
        n: 8 or 4
    Returns:
        Obs (0 for wall, 1 for empty, 2 for goal)
    """
    a = ACTIONS_CARDINAL if hansen_n == 4 else ACTIONS_ORDINAL
    a = a[None, :]
    coords = agent_yx[:, None] + a
    squares = grid[tuple(coords.transpose(2, 0, 1))]
    squares += 1
    squares[squares > 0] = 1  # Empty squares
    if goal_yx is not None:
        is_goal = (goal_yx[:, None] == coords).all(-1)
        squares[is_goal] = 2  # Add goal
    return squares
