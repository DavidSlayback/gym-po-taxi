__all__ = ['grid_to_coord', 'coord_to_grid']

import numpy as np


def grid_to_coord(grid_xy: np.ndarray, cell_size: float = 1.) -> np.ndarray:
    """Convert grid x,y to coordinate xy (middle of given grid square)"""
    return (grid_xy * cell_size) + (cell_size / 2)


def coord_to_grid(coord_xy: np.ndarray, cell_size: float = 1.) -> np.ndarray:
    """Convert x,y coordinate to nearest grid square"""
    return np.round(coord_xy / cell_size).astype(int)