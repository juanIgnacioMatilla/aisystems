import math
from typing import Callable
import numpy as np
from TP4.src.model.kohonen.grid.grid import Grid
from TP4.src.model.kohonen.grid.hex_grid import HexGrid
from TP4.src.model.kohonen.grid.rectangular_grid import RectangularGrid
from TP4.src.model.kohonen.topology import Topology


class SOM:
    def __init__(
            self,
            k: int,
            topology: Topology = Topology.RECTANGULAR,
            learning_rate: Callable[[int], float] = lambda x: 1 / (x + 1.1),
            radius: Callable[[int], int] = None,
    ):
        """
        :param k: Shape of the grid (k x k).
        :param topology: The topology can be 'rectangular' or 'hexagonal'.
        :param learning_rate: Learning rate function per epoch.
        :param radius: Radius function per epoch.
        """
        self.k = k
        self.topology = topology
        self.learning_rate = learning_rate
        self.radius = radius

    def _get_decreasing_radius_func(self, epochs):
        def decreasing_radius(x):
            decay_constant = epochs / math.log(self.k)
            radius = self.k * math.exp(-x / decay_constant)
            return max(1, round(radius))  # Ensure radius stays at least 1
        return decreasing_radius

    def train(self, inputs: np.ndarray, epochs: int):
        if self.radius is None:
            self.radius = self._get_decreasing_radius_func(epochs)
        grid: Grid = initialize_grid(self.k, inputs, self.topology)
        for epoch in range(epochs):
            current_radius = self.radius(epoch)
            current_learning_rate = self.learning_rate(epoch)
            for input_vector in inputs:
                # 1. Find the winning neuron
                bmu = grid.find_bmu(input_vector)

                # 3. Find the neighboring neurons
                neighbors = grid.get_neighbors(bmu, current_radius)

                # 4. Update the weights of the neighboring neurons
                for neuron in neighbors:
                    neuron.update_weights(input_vector, current_learning_rate)
        return grid


def initialize_grid(k: int, inputs: np.ndarray, topology: Topology):
    if topology == Topology.RECTANGULAR:
        return RectangularGrid(k, inputs)
    elif topology == Topology.HEXAGONAL:
        return HexGrid(k, inputs)

