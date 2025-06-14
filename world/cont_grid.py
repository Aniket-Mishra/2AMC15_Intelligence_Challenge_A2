# world/grid.py

import numpy as np

class Grid:
    """
    Provides functions to translate grid coordinates to continuous space and to check obstacles collision
    """
    def __init__(self,
                 cells: np.ndarray,
                 world_size: tuple[float, float],
                 name: str = None):
        """
        :param cells: 2D numpy array of shape (n_rows, n_cols).  Values:
                       0=empty, 1=obstacle, 2=target.
        :param world_size: (width, height)
        """
        assert isinstance(cells, np.ndarray) and cells.ndim == 2
        self.cells = cells
        self.n_rows, self.n_cols = cells.shape

        # world_size = [x_max - x_min, y_max - y_min], e.g. (2 - (-2), 2 - (-2)) = (4,4)
        self.world_width, self.world_height = world_size

        # Precompute how large each cell is in continuous space:
        self.cell_width = self.world_width / self.n_cols
        self.cell_height = self.world_height / self.n_rows

        self.x_min = -self.world_width / 2.0
        self.y_min = -self.world_height / 2.0

        self.name = name

    def continuous_to_cell(self, x: float, y: float) -> tuple[int, int]:
        """
        Convert a continuous coordinate (x,y) into a (row, col) index in `self.cells`.
        If (x,y) is outside the world bounds, we clamp to the nearest cell on the edge.
        """
        # 1) Shift so that x_min maps to 0; then divide by cell_width to get “col index”:
        col = int((x - self.x_min) / self.cell_width)
        row = int((y - self.y_min) / self.cell_height)

        # Clamp:
        col = max(0, min(self.n_cols - 1, col))
        row = max(0, min(self.n_rows - 1, row))
        return row, col

    def is_obstacle(self, x: float, y: float) -> bool:
        """
        Return True if the continuous point (x,y) lies in a cell marked as 1.
        """
        row, col = self.continuous_to_cell(x, y)
        return self.cells[row, col] == 1

    def get_all_obstacle_cells(self) -> list[tuple[int, int]]:
        """
        Returns a list of (row, col) pairs where cells[row,col] == 1.
        """
        idxs = np.argwhere(self.cells == 1)
        return [(int(r), int(c)) for (r, c) in idxs]

    def get_name(self) -> str:
        return self.name
