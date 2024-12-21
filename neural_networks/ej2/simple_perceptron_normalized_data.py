import copy
from abc import ABC

import numpy as np

from neural_networks.src.model.neuron import Neuron


def denormalize_data(y_pred, y_min, y_max):
    return ((y_pred + 1) * (y_max - y_min) / 2) + y_min


class SimplePerceptron(ABC):
    def __init__(self, learning_rate=0.01, activation_function=lambda x: x):
        self.learning_rate = learning_rate
        self.activation_function = activation_function

    def train(self, X, y, y_max, y_min, initial_weights, epochs):
        neuron = Neuron(initial_weights, self.activation_function)
        neurons_by_epoch = [copy.deepcopy(neuron)]
        errors_by_epoch = []
        for epoch in range(epochs):
            epoch_error = 0
            for inputs, target in zip(X, y):
                prediction = neuron.predict(inputs)
                epoch_error += 0.5 * (
                            denormalize_data(target, y_min, y_max) - denormalize_data(prediction, y_min, y_max)) ** 2
                # Update weights and bias
                neuron.weights = self.weights_adjustment(
                    neuron.weights,
                    target,
                    prediction,
                    inputs
                )
                neurons_by_epoch.append(copy.deepcopy(neuron))
            errors_by_epoch.append(epoch_error)
        return neuron, errors_by_epoch, neurons_by_epoch

    def weights_adjustment(self, weights, target, prediction, inputs):
        pass


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

class SimpleLinearPerceptron(SimplePerceptron):

    def weights_adjustment(self, weights, target, prediction, inputs):
        error = target - prediction
        inputs_with_bias = np.append(inputs, 1)
        weights += self.learning_rate * error * inputs_with_bias
        return weights
