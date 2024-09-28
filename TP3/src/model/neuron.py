from typing import Callable

import numpy as np


class Neuron:
    def __init__(self, weights: np.ndarray, activation_function: Callable):
        # Inicializa los pesos aleatorios y un umbral (bias)
        self.weights = weights
        self.activation_function = activation_function

    def predict(self, inputs):
        # Calcula la salida de la neurona
        inputs_with_bias = np.append(inputs, 1)
        linear_combination = np.dot(inputs_with_bias, self.weights)
        return self.activation_function(linear_combination)
