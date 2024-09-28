from abc import ABC

from TP3.src.model.neuron import Neuron


class SimplePerceptron(ABC):
    def __init__(self, learning_rate=0.01, activation_function=lambda x: x):
        self.learning_rate = learning_rate
        self.activation_function = activation_function

    def train(self, X, y, initial_weights, epochs):
        neuron = Neuron(initial_weights, self.activation_function)
        errors_by_epoch = []
        for epoch in range(epochs):
            epoch_error = 0
            for inputs, target in zip(X, y):
                prediction = neuron.predict(inputs)
                epoch_error += 0.5 * (target - prediction) ** 2
                # Update weights and bias
                neuron.weights = self.weights_adjustment(
                    neuron.weights,
                    target,
                    prediction,
                    inputs
                )
            errors_by_epoch.append(epoch_error)
        return neuron, errors_by_epoch

    def weights_adjustment(self, weights, target, prediction, inputs):
        pass
