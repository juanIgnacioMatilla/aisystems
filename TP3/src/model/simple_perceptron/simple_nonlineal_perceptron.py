import numpy as np

from TP3.src.model.neuron import Neuron


class SimpleNonLinearPerceptron:
    def __init__(self, learning_rate):
        self.neuron = None
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, X, y, epochs):
        initial_weights = np.random.rand(len(X[0]) + 1)  # +1 for the bias
        self.neuron = Neuron(initial_weights, self.sigmoid)
        for epoch in range(epochs):
            for inputs, target in zip(X, y):
                prediction = self.neuron.predict(inputs)
                error = target - prediction
                # Update weights and bias
                adjustment = self.learning_rate * error * self.sigmoid_derivative(prediction)
                inputs_with_bias = np.append(inputs, 1)
                self.neuron.weights += adjustment * inputs_with_bias

    def predict(self, inputs):
        return self.neuron.predict(inputs)
