from abc import ABC


class SimplePerceptron(ABC):
    def __init__(self, learning_rate=0.01, activation_function=lambda x: x):
        self.learning_rate = learning_rate
        self.activation_function = activation_function

    def train(self, X, y, initial_weights, epochs):
        pass
