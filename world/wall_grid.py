# world/grid_config.py
from typing import Tuple, Any

import numpy as np
from numpy import ndarray, dtype


def load_grid() -> tuple[ndarray[Any, dtype[Any]], tuple[float, int, float]]:
    """
    Returns a 2D numpy array of integers, where:
      0 = empty cell
      1 = obstacle
      2 = target

    Just an example config, to make a more granular grid just increase the grid size below. A new obstacle or target
     is created by assigning cell values a corresponding number.
    """
    # Example: 20×20 grid
    grid = np.zeros((20, 20), dtype=int)

    # Example vertical line
    for i in range(8):
        grid[i, 10] = 1

    # Example horizontal line
    for i in range(16):
        grid[12, -i] = 1

    starting_position = (1.25, -1, 0.0)

    return grid, starting_position
