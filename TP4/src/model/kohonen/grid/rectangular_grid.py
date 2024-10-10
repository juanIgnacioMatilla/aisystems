from abc import ABC
from typing import Tuple, List

import numpy as np

from TP4.src.model.kohonen.grid.grid import Grid
from TP4.src.model.kohonen.neuron import Neuron


class RectangularGrid(Grid):

    def get_neighbors(self, coords: Tuple[int, int], r: int) -> List[Neuron]:
        row, col = coords
        # Define the bounds for rows and columns to consider
        row_min = max(0, row - r)
        row_max = min(self.size, row + r + 1)
        col_min = max(0, col - r)
        col_max = min(self.size, col + r + 1)
        # Extract the submatrix containing the central element and its neighbors
        submatrix = self.matrix[row_min:row_max, col_min:col_max]
        # Flatten the resulting array to get a 1D list of Neurons
        return submatrix.flatten()
