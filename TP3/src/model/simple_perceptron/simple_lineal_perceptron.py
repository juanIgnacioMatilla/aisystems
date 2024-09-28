from TP3.src.model.neuron import Neuron


class SimpleLinearPerceptron:
    def __init__(self, learning_rate=0.01, activation_function=lambda x: x):
        self.neuron = None
        self.learning_rate = learning_rate
        self.activation_function = activation_function

    def train(self, X, y, epochs):
        self.neuron = Neuron(len(X[0]), self.activation_function)
        for epoch in range(epochs):
            for inputs, target in zip(X, y):
                # Predicción
                prediction = self.neuron.predict(inputs)
                # Cálculo del error
                error = target - prediction

                # Actualiza los pesos y el bias
                self.neuron.weights += self.learning_rate * error * inputs
                self.neuron.bias += self.learning_rate * error

    def predict(self, inputs):
        # Devuelve la salida de la neurona
        return self.neuron.predict(inputs)
