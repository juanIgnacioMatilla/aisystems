import numpy as np

from TP3.src.model.neuron import Neuron
from TP3.src.model.simple_perceptron.simple_perceptron import SimplePerceptron


class SimpleNonLinearPerceptron(SimplePerceptron):
    def __init__(self, learning_rate, activation_function=lambda x: 1 / (1 + np.exp(-x)),
                 activation_function_derivate=lambda x: x * (1 - x)):
        super().__init__(learning_rate, activation_function)
        self.activation_function_derivate = activation_function_derivate

    def train(self, X, y, initial_weights, epochs):
        neuron = Neuron(initial_weights, self.activation_function)
        errors_by_epoch = []
        for epoch in range(epochs):
            epoch_error = 0
            for inputs, target in zip(X, y):
                prediction = neuron.predict(inputs)
                error = target - prediction
                epoch_error += 0.5 * error ** 2
                # Update weights and bias
                adjustment = self.learning_rate * error * self.activation_function_derivate(prediction)
                inputs_with_bias = np.append(inputs, 1)
                neuron.weights += adjustment * inputs_with_bias
            errors_by_epoch.append(epoch_error)
        return neuron, errors_by_epoch
