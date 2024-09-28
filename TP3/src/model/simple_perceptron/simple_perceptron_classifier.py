import numpy as np

from TP3.src.model.neuron import Neuron


class SimplePerceptronClassifier:
    def __init__(self, n_inputs, learning_rate=0.01, activation_function=lambda x: 1 / (1 + np.exp(-x))):
        self.neuron = Neuron(n_inputs, activation_function)
        self.learning_rate = learning_rate

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            for inputs, target in zip(X, y):
                # Prediction
                prediction = self.neuron.predict(inputs)
                # Calculate the error
                error = target - prediction

                # Update weights and bias
                self.neuron.weights += self.learning_rate * error * inputs
                self.neuron.bias += self.learning_rate * error

    def classify(self, inputs):
        # Clasificación binaria (0 o 1)
        output = self.neuron.predict(inputs)
        return 1 if output >= 0.5 else 0