import json

import numpy as np
import pandas as pd

from TP3.src.model.simple_perceptron.simple_lineal_perceptron import (
    SimpleLinearPerceptron,
)
from TP3.src.model.simple_perceptron.simple_nonlineal_perceptron import (
    SimpleNonLinearPerceptron,
)


def set_up(file_path="./config.json"):
    with open(file_path, "r") as file:
        config = json.load(file)
    return config["epochs"], config["learning_rate"], config["p_type"], config["k"]


def new_perceptron(p_type, learning_rate):
    perceptron = {
        "linear": SimpleLinearPerceptron(learning_rate=learning_rate),
        "non_linear": SimpleNonLinearPerceptron(
            learning_rate=learning_rate,
            activation_function=lambda x: 1 / (1 + np.exp(-x)),
            activation_function_derivate=lambda x: x * (1 - x),
        ),
    }
    return perceptron[p_type]


def main():

    # Set up
    epochs, learning_rate, p_type, k = set_up()

    # Load data, specify that the first row is the header
    data = pd.read_csv("../inputs/TP3-ej2-conjunto.csv", header=0)

    # Extract features and target variable
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].astype(float).values  # Convert target to float
    # Normalize the data (optional)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    input_blocks = np.array_split(X, k)
    target_blocks = np.array_split(y, k)
    print(f"Starting k-fold cross validation with k={k}")
    for i in range(k):
        X_train = np.concatenate([input_blocks[j] for j in range(k) if j != i], axis=0)
        y_train = np.concatenate([target_blocks[j] for j in range(k) if j != i], axis=0)
        X_test = input_blocks[i]
        y_test = target_blocks[i]

        # pick perceptron
        perceptron = new_perceptron(p_type, learning_rate)
        # Train the linear perceptron
        initial_weights = np.random.rand(len(X[0]) + 1)  # +1 for the bias
        neuron, errors = perceptron.train(
            X_train, y_train, initial_weights, epochs=epochs
        )
        # Validate the perceptron
        print("Perceptron: ", p_type)
        test_error = 0.0
        for inputs, expected in zip(X_test, y_test):
            prediction = neuron.predict(inputs)
            print(
                f"Entrada: {inputs}, Predicción: {prediction:.3f}, Esperado: {expected}"
            )
            test_error += 0.5 * (expected - prediction) ** 2
            print()
        for i, error in enumerate(errors):
            if i % 5000 == 0:
                print("error for epoch ", i, ": ", error)
        print("final error: ", errors[-1])
        print("mean squared error: ", errors[-1] / len(X_train))
        print("test error: ", test_error / len(X_test))
        print()
    #
    # # Extract features and target variable
    # X = data.iloc[:, :-1].values
    # y = data.iloc[:, -1].astype(float).values  # Convert target to float
    #
    # # Train the nonlinear perceptron
    # nonlinear_perceptron = SimpleNonLinearPerceptron(
    #     learning_rate=0.001,
    #     activation_function=lambda x: 1 / (1 + np.exp(-x)),
    #     activation_function_derivate=lambda x: x * (1 - x),
    # )
    #
    # initial_weights = np.random.uniform(
    #     -0.01, 0.01, size=(len(X[0]) + 1)
    # )  # Use a smaller range
    # nonlinear_neuron, errors = nonlinear_perceptron.train(
    #     X, y, initial_weights, epochs=30000
    # )
    # # Validate the perceptron
    # print("Nonlinear Perceptron")
    # for inputs, expected in zip(X, y):
    #     print(
    #         f"Entrada: {inputs}, Predicción: {nonlinear_neuron.predict(inputs)}, Esperado: {expected}"
    #     )
    # print()
    # for i, error in enumerate(errors):
    #     if i % 5000 == 0:
    #         print("error for epoch ", i, ": ", error)


if __name__ == "__main__":
    main()
