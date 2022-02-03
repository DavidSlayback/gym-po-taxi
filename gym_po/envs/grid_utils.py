import numpy as np


# Standard 2D directions (typical for actions)
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

# Standard 3D directions (for multistory)
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


def get_surrounding_indices(coordinate: np.ndarray, surround: int = 1) -> np.ndarray:
    """Returns indices for all coordinates surrounding some coordinate (useful for highlighting)

    Args:
        coordinate: Coordinate(s) in shape (n_coord?, n_dim)
        surround: Number of squares out to look
    """
    if not surround: return coordinate
    coordinate = np.atleast_2d(coordinate)  #n_coord, n_dim
    # assert (obs_n % 2) == 1
    ncoord, ndim = coordinate.shape
    # np mgrid generates a mesh of nDimxd1xd2xd3...
    if ndim == 2:
        g = np.mgrid[-surround:surround+1, -surround:surround+1]
        g = g.reshape(-1, 2)  # unroll to same format as coordinate. mgrid[:3,:3] unrolls to 9x2
        ac = coordinate[..., None, None] + g
        ac = ac.reshape(ncoord, -1, 2)
    else:
        g = np.mgrid[:1, -surround:surround+1, -surround:surround+1]
        g = g.reshape(-1, 3)
        ac = coordinate[..., None, None, None] + g
        ac = ac.reshape(ncoord, -1, 3)
    return ac

if __name__ == "__main__":
    tcoord = np.array([[2,2], [4,4]])
    test = get_surrounding_indices(tcoord, 1)
    print(3)