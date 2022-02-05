from typing import Sequence, Tuple, Union, Callable, Iterable
from functools import partial
import numpy as np


# Standard 2D (yx) directions (typical for actions)
class DIRECTIONS_2D:
    up = north = np.array([-1, 0], dtype=int)
    down = south = np.array([1, 0], dtype=int)
    left = west = np.array([0, -1], dtype=int)
    right = east = np.array([0, 1], dtype=int)
    northwest = np.array([-1, -1], dtype=int)
    northeast = np.array([-1, 1], dtype=int)
    southwest = np.array([1, -1], dtype=int)
    southeast = np.array([1, 1], dtype=int)

DIRECTIONS_2D_NP = np.array([DIRECTIONS_2D.up, DIRECTIONS_2D.down, DIRECTIONS_2D.west, DIRECTIONS_2D.east,
                      DIRECTIONS_2D.northwest, DIRECTIONS_2D.northeast, DIRECTIONS_2D.southwest,
                      DIRECTIONS_2D.southeast]).T

# Standard 3D (zyx) directions (for multistory)
class DIRECTIONS_3D:
    up = north = np.array([0, -1, 0], dtype=int)
    down = south = np.array([0, 1, 0], dtype=int)
    left = west = np.array([0, 0, -1], dtype=int)
    right = east = np.array([0, 0, 1], dtype=int)
    northwest = np.array([0, -1, -1], dtype=int)
    northeast = np.array([0, -1, 1], dtype=int)
    southwest = np.array([0, 1, -1], dtype=int)
    southeast = np.array([0, 1, 1], dtype=int)
    upstairs = np.array([1, 0, 0], dtype=int)
    downstairs = np.array([-1, 0, 0], dtype=int)


WALLS = {'|', '-'}  # Vertical and horizontal walls


def get_surrounding_indices(coordinate: np.ndarray, surround: int = 1) -> np.ndarray:
    """Returns indices for all coordinates surrounding some coordinate (useful for highlighting)

    Args:
        coordinate: Coordinate(s) in shape (n_dim, n_coord?)
        surround: Number of squares out to look
    Returns:
        coordinates: Cooordinates (ndim, ncoord, n_surround_coord_per_coord), can be reshaped to (ndim, n_all_coord)
    """
    if not surround: return coordinate
    if coordinate.ndim == 1: coordinate = coordinate.reshape(coordinate.shape + (1,))  # make at least 2d
    ndim, ncoord = coordinate.shape
    # np mgrid generates a mesh of nDimxd1xd2xd3...
    if ndim == 2: g = np.mgrid[-surround:surround+1, -surround:surround+1]
    else: g = np.mgrid[:1, -surround:surround+1, -surround:surround+1]
    g = g.reshape(ndim, -1)  # unroll to same format as coordinate. mgrid[:3,:3] unrolls to 9x2
    g = g[:, (g[-2:] != 0).any(0)]  # Remove center coordinate
    ac = (g[:, None] + coordinate[...,None]).reshape(ndim, ncoord, -1)
    return ac


def get_hansen_indices(coordinate: np.ndarray) -> np.ndarray:
    """Returns indices for cardinally-adjacent coordinates (useful for highlighting)

    Args:
        coordinate: Coordinate(s) in shape (n_dim, n_coord?)
    Returns:
        coordinates: Cooordinates (ndim, ncoord, 4), can be reshaped to (ndim, 4 * n_coord)
    """
    if coordinate.ndim == 1: coordinate = coordinate.reshape(coordinate.shape + (1,))  # make at least 2d
    ndim, ncoord = coordinate.shape
    g = [[-1, 1, 0, 0], [0, 0, -1, 1]]  # N, S, W, E
    for d in range(3-ndim): g.insert(0, [0,0,0,0])
    ac = (g[:, None] + coordinate[...,None]).reshape(ndim, ncoord, -1)
    return ac


def get_flat_to_coord_function(grid_shape: Sequence[int]) -> Callable[[Union[Iterable, np.ndarray, int, float]], np.ndarray]:
    """Returns a partial np.unravel_index to convert (z)yx coordinates to single integers

    Args:
        grid_shape: Corresponding shape of grid to do coordinates for
    Return:
        fn: Callable that converts 1 or more flat coordinates to grid coordinates (ndimxncoord)
    """
    fn = partial(np.unravel_index, shape=grid_shape)
    def f(flat_coordinates: Union[Iterable, np.ndarray, int, float]) -> np.ndarray: return np.array(fn(flat_coordinates))
    return f


def get_coord_to_flat_function(grid_shape: Sequence[int]) -> Callable[[Union[Iterable, Tuple[np.ndarray], int, float]], np.ndarray]:
    """Returns a partial np.unravel_index to convert (z)yx coordinates to single integers

    Args:
        grid_shape: Corresponding shape of grid to do coordinates for
    Return:
        fn: Callable that converts 1 or more grid coordinates (ndimxncoord) to flat coordinates (ncoord)
    """
    fn = partial(np.ravel_multi_index, dims=grid_shape, mode='wrap')
    return fn


if __name__ == "__main__":
    tcoord = np.array([[2,2], [4,4], [8,8]]).T
    tcoord3 = np.array([[0, 2, 2], [1, 4, 4], [2, 8, 8]]).T
    test = get_surrounding_indices(tcoord3, 1)
    print(3)