import numpy as np

from neural_networks.src.model.multilayer_perceptron.momentum.momentum_layer import MomentumLayer
from neural_networks.src.model.multilayer_perceptron.vanilla.multi_layer_perceptron import MultiLayerPerceptron


class MomentumMultiLayerPerceptron(MultiLayerPerceptron):
    def __init__(self, layers_structure, learning_rate=0.01, alpha=0.9,
                 activation_function=lambda x: 1 / (1 + np.exp(-x)),
                 activation_function_derivative=lambda x: x * (1 - x)):
        super().__init__(layers_structure, learning_rate, activation_function, activation_function_derivative)
        self.alpha = alpha
        self.layers = [
            MomentumLayer(num_neurons, input_size, activation_function, activation_function_derivative, alpha)
            for input_size, num_neurons in zip(layers_structure[:-1], layers_structure[1:])
        ]
