import numpy as np


class Neuron:
    def __init__(self, weights: np.ndarray):
        self.weights = weights

    def update_weights(self, input_vector: np.ndarray, learning_rate: float):

        self.weights += learning_rate * (input_vector - self.weights)

    def __repr__(self) -> str:
        return f"Neuron(weights: {self.weights})"
