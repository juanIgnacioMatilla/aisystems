import numpy as np

from neural_networks.src.model.neuron import Neuron


class Layer:
    def __init__(
            self,
            num_neurons,
            input_size,
            activation_function,
            activation_function_derivative
    ):
        self.neurons = [
            Neuron(np.random.randn(input_size + 1), activation_function) for _ in range(num_neurons)
        ]
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative

    def predict(self, inputs):
        return np.array([neuron.predict(inputs) for neuron in self.neurons])

    def adjust_weights(self, inputs, deltas, learning_rate):
        for neuron, delta in zip(self.neurons, deltas):
            inputs_with_bias = np.append(inputs, 1)
            neuron.weights += learning_rate * delta * inputs_with_bias

    def get_weights_matrix(self):
        # Collect weights (excluding bias term) from all neurons as a matrix
        return np.array([neuron.weights[:-1] for neuron in self.neurons])
