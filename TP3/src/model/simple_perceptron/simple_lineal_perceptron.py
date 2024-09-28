from TP3.src.model.neuron import Neuron


class SimpleLinearPerceptron:
    def __init__(self, n_inputs, learning_rate=0.01):
        self.neuron = Neuron(n_inputs, lambda x: x)
        self.learning_rate = learning_rate

    def train(self, X, y, epochs):
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
