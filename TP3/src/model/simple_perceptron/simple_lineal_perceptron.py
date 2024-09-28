import numpy as np

from TP3.src.model.neuron import Neuron
from TP3.src.model.simple_perceptron.simple_perceptron import SimplePerceptron


class SimpleLinearPerceptron(SimplePerceptron):
    def train(self, X, y, initial_weights, epochs):
        neuron = Neuron(initial_weights, self.activation_function)
        errors_by_epoch = []
        for epoch in range(epochs):
            epoch_error = 0  # To accumulate the error over all samples
            for inputs, target in zip(X, y):
                prediction = neuron.predict(inputs)
                error = target - prediction
                epoch_error += 0.5 * error ** 2
                # Update weights and bias
                inputs_with_bias = np.append(inputs, 1)
                neuron.weights += self.learning_rate * error * inputs_with_bias
            errors_by_epoch.append(epoch_error)
        return neuron, errors_by_epoch
