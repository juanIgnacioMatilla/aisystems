from abc import ABC
from typing import Tuple, List, Callable
import numpy as np
from TP4.src.model.kohonen.neuron import Neuron
from TP4.src.propagation_funcs.min_euclidean_distance import min_euclidean_distance


class Grid(ABC):
    def __init__(
            self,
            size: int,
            weights: np.ndarray,
            propagation_func: Callable[[int, np.ndarray, np.ndarray], Tuple[int, int]] = min_euclidean_distance
    ):
        self.size: int = size
        # Create a matrix of Neurons and ensure it's a NumPy array
        self.matrix: np.ndarray = np.empty((size, size), dtype=Neuron)
        # Shuffle the weights along the first axis
        np.random.shuffle(weights)
        # Create a matrix of Neurons using the shuffled weights
        for j in range(size):
            for i in range(size):
                # Standardize weights
                self.matrix[j, i] = Neuron((weights[j + i] - np.mean(weights) / np.std(weights)))
        self.propagation_func = propagation_func

    def find_bmu(self, input_vector: np.ndarray) -> Tuple[int, int]:
        """
        Finds the Best Matching Unit (BMU) in the grid.
        Parameters:
        - input_vector: The input vector to compare with the neurons' weights.
        Returns:
        - Tuple[int, int]: Coordinates of the BMU in the grid.
        """
        return self.propagation_func(self.size, self.matrix, np.array(input_vector))

    def get_neighbors(self, coords: Tuple[int, int], r) -> List[Neuron]:
        """
        Finds the neighbors of an element in the matrix given a radius r.
        Parameters:
        - row: index of the row of the central element.
        - col: index of the column of the central element.
        - r: radius to consider for neighbors.
        """
        pass
