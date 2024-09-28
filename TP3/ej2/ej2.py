import numpy as np
import pandas as pd
from TP3.src.model.simple_perceptron.simple_lineal_perceptron import SimpleLinearPerceptron
from TP3.src.model.simple_perceptron.simple_nonlineal_perceptron import SimpleNonLinearPerceptron

def main():
    # Load data, specify that the first row is the header
    data = pd.read_csv("../inputs/TP3-ej2-conjunto.csv", header=0)

    # Extract features and target variable
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].astype(float).values  # Convert target to float
    # Normalize the data (optional)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    # Train the linear perceptron
    linear_perceptron = SimpleLinearPerceptron(learning_rate=0.00001)
    initial_weights = np.random.rand(len(X[0]) + 1)  # +1 for the bias
    linear_neuron, errors = linear_perceptron.train(X, y, initial_weights, epochs=30000)
    # Validate the perceptron
    print("Linear Perceptron")
    for inputs, expected in zip(X,y):
        print(f"Entrada: {inputs}, Predicción: {linear_neuron.predict(inputs):.3f}, Esperado: {expected}")
    print()
    for i, error in enumerate(errors):
        if i % 5000 == 0:
            print("error for epoch ", i, ": ", error)
    print()


    # Extract features and target variable
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].astype(float).values  # Convert target to float

    # Train the nonlinear perceptron
    nonlinear_perceptron = SimpleNonLinearPerceptron(
        learning_rate=0.001,
        activation_function=lambda x:  1 / (1 + np.exp(-np.clip(x, -500, 500))),
        activation_function_derivate=lambda x: (x * (1 - x))
    )

    initial_weights = np.random.uniform(-0.01, 0.01, size=(len(X[0]) + 1))  # Use a smaller range
    nonlinear_neuron, errors = nonlinear_perceptron.train(X, y, initial_weights, epochs=30000)
    # Validate the perceptron
    print("Nonlinear Perceptron")
    for inputs, expected in zip(X, y):
        print(f"Entrada: {inputs}, Predicción: {nonlinear_neuron.predict(inputs)}, Esperado: {expected}")
    print()
    for i, error in enumerate(errors):
        if i % 5000 == 0:
            print("error for epoch ", i, ": ", error)


if __name__ == "__main__":
    main()
