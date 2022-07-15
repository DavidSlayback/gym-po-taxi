__all__ = ['grid_to_coord', 'coord_to_grid']

import numpy as np


def grid_to_coord(grid_yx: np.ndarray, cell_size: float = 1.) -> np.ndarray:
    """Convert grid y,x to coordinate xy (middle of given grid square)"""
    return (grid_yx * cell_size) + (cell_size / 2)


def coord_to_grid(coord_yx: np.ndarray, cell_size: float = 1.) -> np.ndarray:
    """Convert y,x coordinate to nearest grid square"""
    return np.round(coord_yx / cell_size).astype(int)