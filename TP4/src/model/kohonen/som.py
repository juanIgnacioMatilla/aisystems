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
            topology: Topology,
            learning_rate: Callable[[int], float],
            radius: Callable[[int], int],
    ):
        """
        :param k: Forma de la cuadrícula (k x k).
        :param topology: La topología puede ser 'rectangular' o 'hexagonal'.
        :param learning_rate: Funcion de taza de aprendizaje por epoca.
        :param radius: Funcion de radio por epoca.
        """
        self.k = k
        self.topology = topology
        self.learning_rate = learning_rate
        self.radius = radius

    def train(self, inputs: np.ndarray, epochs: int):
        grid: Grid = initialize_grid(self.k, inputs, self.topology)

        inputs_per_neuron_by_epoch = []
        for epoch in range(epochs):
            current_radius = self.radius(epoch)
            current_learning_rate = self.learning_rate(epoch)
            # Create a dictionary to store the inputs assigned to each neuron
            inputs_per_neuron = {(i, j): [] for i in range(self.k) for j in range(self.k)}

            for input_vector in inputs:
                # 1. Encuentra la neurona ganadora
                bmu = grid.find_bmu(input_vector)

                # 2. Almacena el input en la neurona ganadora
                inputs_per_neuron[bmu].append(input_vector)

                # 3. Encuentra las neuronas vecinas
                neighbors = grid.get_neighbors(bmu, current_radius)

                # 4. Actualiza los pesos de las neuronas en el vecindario
                for neuron in neighbors:
                    neuron.update_weights(input_vector, current_learning_rate)
            inputs_per_neuron_by_epoch.append(inputs_per_neuron)
        return grid, inputs_per_neuron_by_epoch


def initialize_grid(k: int, inputs: np.ndarray, topology: Topology):
    if topology == Topology.RECTANGULAR:
        return RectangularGrid(k, inputs)
    elif topology == Topology.HEXAGONAL:
        return HexGrid(k, inputs)
