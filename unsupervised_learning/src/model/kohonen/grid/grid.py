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
        self.propagation_func = propagation_func
        self.size: int = size
        # Create a matrix of Neurons and ensure it's a NumPy array
        self.matrix: np.ndarray = np.empty((size, size), dtype=Neuron)
        # Determine the range from the inputs
        min_input, max_input = np.min(weights), np.max(weights)
        # Create a matrix of Neurons using the shuffled weights
        for j in range(size):
            for i in range(size):
                random_weight = np.random.uniform(min_input, max_input, size=weights[0].shape)
                self.matrix[j, i] = Neuron(random_weight)


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

    def calculate_average_distances(self):
        k = self.size
        average_distances = np.zeros((k, k))

        for i in range(k):
            for j in range(k):
                # Obtener el vector de pesos de la neurona actual
                current_weights = self.matrix[i, j].weights

                # Obtener las neuronas vecinas
                neighbors = self.get_neighbors((i, j), 1)
                print(neighbors, 'neighbors', i, j)

                distances = []

                for n in neighbors:
                    # Obtener el vector de pesos de la neurona vecina
                    neighbor_weights = n.weights

                    # Calcular la distancia euclidiana entre la neurona actual y la neurona vecina
                    distance = np.linalg.norm(current_weights - neighbor_weights)

                    # Agregar la distancia calculada a la lista de distancias
                    distances.append(distance)

                average_distances[i, j] = np.mean(distances) if distances else 0  # Evitar la división por cero

        return average_distances
