__all__ = ["grid_to_coord", "coord_to_grid"]

import numpy as np
from numpy.typing import NDArray


def grid_to_coord(grid_yx: NDArray[int], cell_size: float = 1.0) -> NDArray[float]:
    """Convert grid y,x to coordinate xy (middle of given grid square)

    Places agent in middle of grid square
    """
    return (grid_yx * cell_size) + (cell_size / 2)


def coord_to_grid(coord_yx: NDArray[float], cell_size: float = 1.0) -> NDArray[int]:
    """Convert y,x coordinate to nearest grid square.

    Uses floor because anything from 0-1 is in square '0'
    """
    return np.floor(coord_yx / cell_size).astype(int)
