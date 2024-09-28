import numpy as np
from TP3.src.model.simple_perceptron.simple_perceptron import SimplePerceptron


class SimpleNonLinearPerceptron(SimplePerceptron):
    def __init__(self, learning_rate, activation_function=lambda x: 1 / (1 + np.exp(-x)),
                 activation_function_derivate=lambda x: x * (1 - x)):
        super().__init__(learning_rate, activation_function)
        self.activation_function_derivate = activation_function_derivate

    def weights_adjustment(self, weights, target, prediction, inputs):
        error = target - prediction
        adjustment = self.learning_rate * error * self.activation_function_derivate(prediction)
        inputs_with_bias = np.append(inputs, 1)
        weights += adjustment * inputs_with_bias
        return weights
