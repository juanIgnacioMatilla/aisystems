import numpy as np

from TP3.src.model.simple_perceptron.simple_lineal_perceptron import SimpleLinearPerceptron
from TP3.src.model.simple_perceptron.simple_nonlineal_perceptron import SimpleNonLinearPerceptron
from TP3.src.model.simple_perceptron.simple_perceptron_classifier import SimplePerceptronClassifier

# Ejemplo de uso
if __name__ == "__main__":

    #Clafisicador
    # Datos de entrenamiento (AND lógico)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])  # Salidas correspondientes para el AND

    perceptron = SimplePerceptronClassifier(n_inputs=2, learning_rate=0.1)
    perceptron.train(X, y, epochs=10)

    for inputs in X:
        print(f"Entrada: {inputs}, Clasificación: {perceptron.classify(inputs)}")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 2])
    print()

    #Lineal
    perceptron = SimpleLinearPerceptron(n_inputs=2, learning_rate=0.1)
    perceptron.train(X, y, epochs=100)
    for inputs in X:
        print(f"Entrada: {inputs}, Predicción: {perceptron.predict(inputs)}")
    print()

    #No lineal
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])  

    perceptron = SimpleNonLinearPerceptron(n_inputs=2)
    perceptron.train(X, y, epochs=10000)
    for inputs in X:
        print(f"Entrada: {inputs}, Predicción: {perceptron.predict(inputs)}")
    print()