import numpy as np

from TP3.src.model.neuron import Neuron
from TP3.src.model.simple_perceptron.simple_lineal_perceptron import SimpleLinearPerceptron
from TP3.src.model.simple_perceptron.simple_nonlineal_perceptron import SimpleNonLinearPerceptron
from TP3.src.model.simple_perceptron.simple_lineal_perceptron_classifier import SimplePerceptronClassifier

if __name__ == "__main__":

    # Calsificador Lineal
    # Datos de entrenamiento (AND lógico)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])  # Salidas correspondientes para el AND
    initial_weights = np.random.rand(len(X[0]) + 1)  # +1 for the bias

    perceptron_classifier = SimplePerceptronClassifier(learning_rate=0.1)
    trained_neuron, errors_by_epoch = perceptron_classifier.train(X, y, initial_weights, epochs=100)

    for inputs in X:
        print(f"Entrada: {inputs}, Clasificación: {trained_neuron.predict(inputs)}")
    print()

    # Lineal
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 2])
    initial_weights = np.random.rand(len(X[0]) + 1)  # +1 for the bias
    linear_perceptron = SimpleLinearPerceptron(learning_rate=0.1)
    trained_neuron, errors_by_epoch = linear_perceptron.train(X, y, initial_weights, epochs=100)
    for inputs in X:
        print(f"Entrada: {inputs}, Predicción: {trained_neuron.predict(inputs)}")
    print()

    # No lineal
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])
    initial_weights = np.random.rand(len(X[0]) + 1)  # +1 for the bias

    nonlineal_perceptron = SimpleNonLinearPerceptron(learning_rate=0.01)
    trained_neuron, errors_by_epoch = nonlineal_perceptron.train(X, y,initial_weights, epochs=10000)
    for inputs in X:
        print(f"Entrada: {inputs}, Predicción: {trained_neuron.predict(inputs)}")
    print()
