import numpy as np

from TP3.src.model.multilayer_perceptron.adam.adam_layer import AdamLayer
from TP3.src.model.multilayer_perceptron.vanilla.multi_layer_perceptron import MultiLayerPerceptron


class AdamMultiLayerPerceptron(MultiLayerPerceptron):
    def __init__(self, layers_structure, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 activation_function=lambda x: 1 / (1 + np.exp(-x)),
                 activation_function_derivative=lambda x: x * (1 - x)):
        super().__init__(layers_structure, learning_rate, activation_function, activation_function_derivative)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.layers = [
            AdamLayer(
                num_neurons,
                input_size,
                activation_function,
                activation_function_derivative,
                beta1=beta1,
                beta2=beta2,
                epsilon=epsilon
            )
            for input_size, num_neurons in zip(layers_structure[:-1], layers_structure[1:])
        ]
