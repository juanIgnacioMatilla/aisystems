from abc import ABC
from typing import Tuple, List

import numpy as np

from TP4.src.model.kohonen.grid.grid import Grid
from TP4.src.model.kohonen.neuron import Neuron


class RectangularGrid(Grid):

    def get_neighbors(self, coords: Tuple[int, int], r: int) -> List[Neuron]:
        neurons = []
        for i in range(self.size):
            for j in range(self.size):
                if i == coords[0] and j == coords[1]:
                    continue
                dist_to_bmu = self._euclidean_distance(np.array([i, j]), np.array(coords))
                if dist_to_bmu <= r:
                    neurons.append(self.matrix[i, j])
        return neurons

    def _euclidean_distance(self, x, y):
        return np.linalg.norm(x - y)