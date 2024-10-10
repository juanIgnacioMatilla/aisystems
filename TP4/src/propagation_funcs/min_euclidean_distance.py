from typing import Tuple

import numpy as np


def min_euclidean_distance(size:int,matrix: np.ndarray , input_vector: np.ndarray) -> Tuple[int, int]:
    min_dist = float('inf')
    bmu = None
    for i in range(size):
        for j in range(size):
            neuron = matrix[i][j]
            # Calculate the Euclidean distance between the input vector and the neuron's weights
            dist = np.linalg.norm(input_vector - neuron.weights)
            if dist < min_dist:
                min_dist = dist
                bmu = (i, j)
    return bmu