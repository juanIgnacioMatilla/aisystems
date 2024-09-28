import numpy as np

from TP3.src.model.neuron import Neuron
from TP3.src.model.simple_perceptron.simple_perceptron import SimplePerceptron


class SimpleLinearPerceptron(SimplePerceptron):

    def weights_adjustment(self, weights, target, prediction, inputs):
        error = target - prediction
        inputs_with_bias = np.append(inputs, 1)
        weights += self.learning_rate * error * inputs_with_bias
        return weights
