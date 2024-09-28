import numpy as np

from TP3.src.model.neuron import Neuron


class SimpleNonLinearPerceptron:
    def __init__(self, learning_rate):
        # Usamos la función sigmoide como función de activación no lineal
        self.neuron = None
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        # Función sigmoide
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # Derivada de la función sigmoide
        return x * (1 - x)

    def train(self, X, y, epochs):
        self.neuron = Neuron(len(X[0]), self.sigmoid)
        for epoch in range(epochs):
            for inputs, target in zip(X, y):
                # Predicción
                prediction = self.neuron.predict(inputs)
                # Cálculo del error
                error = target - prediction
                # Actualiza los pesos y el bias usando la derivada de la sigmoide
                adjustment = self.learning_rate * error * self.sigmoid_derivative(prediction)
                inputs_with_bias = np.append(inputs, 1)
                self.neuron.weights += adjustment * inputs_with_bias

    def predict(self, inputs):
        # Devuelve la salida de la neurona
        return self.neuron.predict(inputs)