import numpy as np
from neural_networks.src.model.multilayer_perceptron.vanilla.layer import Layer
from neural_networks.src.model.neuron import Neuron


class MomentumLayer(Layer):
    def __init__(self, num_neurons, input_size, activation_function, activation_function_derivative, alpha):
        super().__init__(num_neurons, input_size, activation_function, activation_function_derivative)
        self.alpha = alpha
        self.previous_weight_updates = [np.zeros_like(neuron.weights) for neuron in self.neurons]

    def adjust_weights(self, inputs, deltas, learning_rate):
        for i, (neuron, delta) in enumerate(zip(self.neurons, deltas)):
            inputs_with_bias = np.append(inputs, 1)

            # Calcular la actualización de pesos usando la tasa de aprendizaje y el delta
            weight_update = learning_rate * delta * inputs_with_bias

            # Aplicar Momentum usando la actualización previa
            neuron.weights += weight_update + self.alpha * self.previous_weight_updates[i]

            # Guardar la actualización actual como la anterior para la próxima iteración
            self.previous_weight_updates[i] = weight_update
