# world/grid_config.py
from typing import Tuple, Any

import numpy as np
from numpy import ndarray, dtype


def load_grid() -> tuple[ndarray[Any, dtype[Any]], tuple[float, float, float]]:
    """
    Returns a 2D numpy array of integers, where:
      0 = empty cell
      1 = obstacle
      2 = target

    Just an example config, to make a more granular grid just increase the grid size below. A new obstacle or target
     is created by assigning cell values a corresponding number.
    """
    # Example: 20×20 grid
    grid = np.zeros((8, 8), dtype=int)

    grid[1, 0] = 1
    grid[3, 0] = 1
    grid[5, 0] = 1
    grid[7, 0] = 1

    grid[0, 7] = 1
    grid[2, 7] = 1
    grid[0, 5] = 1
    grid[0, 3] = 1

    grid[6, 7] = 1
    grid[4, 7] = 1
    grid[6, 3] = 1
    grid[4, 3] = 1

    starting_position = (-0.75, -0.75, 0.0)

    return grid, starting_position
