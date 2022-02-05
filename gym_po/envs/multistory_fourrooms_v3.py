from typing import Sequence, Tuple, Callable
import numpy as np
from .grid_utils import WALLS

# Fourrooms minus the external wall (13x13 -> 11x11)
# Upstairs and downstairs locations marked by "U" and "D"
BASE_FOURROOMS_MAP_WITH_STAIRS = (
    "     |    U",
    "     |     ",
    "           ",
    "     |     ",
    "     |     ",
    "- ---|     ",
    "     |-- --",
    "     |     ",
    "     |     ",
    "           ",
    "D    |     ",
)


def convert_str_map_to_walled_np_str(map: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, Callable]:
    """Return the full map (for image), the reduced map (for navigation), and a callable to convert coordinates from the 2nd to the first"""
    bordered_map = np.pad(np.asarray(map, dtype='c').astype(str), 1, constant_values='|')
    return bordered_map, bordered_map[1:-1, 1:-1], lambda r,c: (r+1, c+1)


def compute_obs_space(layout: np.ndarray, hansen: bool = False) -> int:
    """Compute discrete observation space

    Args:
        layout: Full map (floors, y, x), can be bordered or now
        hansen: Use hansen (adjacent walls only) observations
    Returns:
        n: Number of possible discrete observations
    """
    return int(2 ** 4) if hansen else int((~np.isin(layout, WALLS)).sum().item())

if __name__ == "__main__":
    bmap, tmap, cc = convert_str_map_to_walled_np_str(BASE_FOURROOMS_MAP_WITH_STAIRS)