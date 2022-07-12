__all__ = ['get_hansen_obs', 'get_number_discrete_states_and_conversion', 'get_number_abstract_states', 'get_grid_obs']
from typing import Tuple
import numpy as np
from .actions import ACTIONS_CARDINAL, ACTIONS_ORDINAL

def get_number_discrete_states_and_conversion(grid: np.ndarray) -> Tuple[int, np.ndarray]:
    """Count the number of possible state observation given a gridified version of a layout

    Args:
        grid: Gridified layout
    Returns:
        n_state: Number of discrete states
        state_grid: Grid that converts (y,x) coords to discrete state
    """
    n_states = (grid >= 0).sum()
    state_grid = (grid >= 0).cumsum()
    return n_states, state_grid


def get_number_abstract_states(grid: np.ndarray) -> int:
    """Count number of rooms

    Args:
        grid: Gridified layout
    Returns:
        n_state: Number of abstract "room" states
    """
    n_states = np.unique(grid) - 1  # Ignore walls
    return n_states


def get_hansen_obs(agent_yx: np.ndarray, grid: np.ndarray, goal_yx: np.ndarray, hansen_n: int = 8) -> int:
    """Get hansen observation of agent(s) (empty, wall, goal) based on grid

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
    is_goal = (goal_yx[:, None] == coords).all(-1)
    squares = grid[tuple(coords.transpose(2,0,1))]
    squares += 1
    squares[squares > 0] = 1  # Empty squares
    squares[is_goal] = 2  # Add goal
    multipliers = np.array([2 ** i for i in range(a.shape[1])])  # There's only one goal, let's let it alias with other possibilities
    return squares.dot(multipliers)


def get_grid_obs(agent_yx: np.ndarray, grid: np.ndarray, goal_yx: np.ndarray, n: int = 3) -> np.ndarray:
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
    coords = agent_yx.swapaxes(0,1)[...,None, None] + mg[None, ...]
    # All invalid coords should point to a wall (i.e., (0,0))
    coords[coords < 0] = 0
    coords[:, (coords[0] >= grid.shape[0]) | (coords[1] >= grid.shape[1])] = 0
    is_goal = (goal_yx.swapaxes(0,1)[..., None, None] == coords).all(0)
    squares = grid[tuple(coords)] + 1
    squares[squares > 0] = 1
    squares[is_goal] = 2
    return squares


