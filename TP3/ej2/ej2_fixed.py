import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from TP3.ej2.simple_perceptron_normalized_data import SimpleLinearPerceptron, SimpleNonLinearPerceptron


def set_up(file_path="./config.json"):
    with open(file_path, "r") as file:
        config = json.load(file)
    return config["epochs"], config["learning_rate"], config["p_type"], config["k"], config["runs"]


def new_perceptron(p_type, learning_rate):
    perceptron = {
        "linear": SimpleLinearPerceptron(learning_rate=learning_rate),
        "non_linear": SimpleNonLinearPerceptron(
            learning_rate=learning_rate,
            activation_function=tanh,
            activation_function_derivate=tanh_deriv,
        ),
    }
    return perceptron[p_type]


# Función tanh
def tanh(x):
    return np.tanh(x)


# Derivada de tanh
def tanh_deriv(x):
    return 1 - np.tanh(x) ** 2


def normalize_data(y, y_min, y_max):
    normalized_y = []
    for target in y:
        normalized_y.append((2 * (target - y_min) / (y_max - y_min)) - 1)
    return np.array(normalized_y)


def denormalize_data(y_pred, y_min, y_max):
    return ((y_pred + 1) * (y_max - y_min) / 2) + y_min


def train_test_split(X, y, test_size=0.25):
    """
    Split data into training and test sets based on the test_size ratio.
    """
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split_idx = int(len(X) * (1 - test_size))

    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test


def main(test_size=0.25):  # Changed to test_size
    epochs, learning_rate, p_type, k_folds, runs = set_up()
    # Load data, specify that the first row is the header
    data = pd.read_csv("../inputs/TP3-ej2-conjunto.csv", header=0)
    # Extract features and target variable
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].astype(float).values  # Convert target to float
    y_min, y_max = y.min(), y.max()
    y = normalize_data(y, y_min=y_min, y_max=y_max)

    all_errors = []  # List to store errors from each run
    all_tests_errors = []

    for run in range(runs):
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        fold_errors = []  # Store errors for each run
        fold_test_errors = []

        perceptron = new_perceptron(p_type, learning_rate)

        initial_weights = np.random.uniform(-0.1, 0.1, size=(len(X_train[0]) + 1))

        nonlinear_neuron, errors, neurons = perceptron.train(
            X_train, y_train, y_max, y_min, initial_weights, epochs=epochs
        )

        tests_errors_by_epoch = []
        for epoch in range(epochs):
            epoch_error = 0
            for inputs, target in zip(X_test, y_test):
                prediction = neurons[epoch].predict(inputs)
                epoch_error += 0.5 * (
                            denormalize_data(target, y_min, y_max) - denormalize_data(prediction, y_min, y_max)) ** 2
            tests_errors_by_epoch.append(epoch_error)

        fold_errors.append(errors)  # Store errors for this run
        fold_test_errors.append(tests_errors_by_epoch)
        all_errors.append(np.mean(fold_errors, axis=0))  # Average errors
        all_tests_errors.append(np.mean(fold_test_errors, axis=0))

    # Convert to numpy array for calculations
    all_errors = np.array(all_errors)
    all_tests_errors = np.array(all_tests_errors)
    print(f"Mean training error las epoch:  {np.mean(all_errors[:-1])} +- {np.std(all_errors[:-1])}")

    print(f"Mean training error las epoch:  {np.mean(all_errors[:-1])} +- {np.std(all_errors[:-1])}")
    # Calculate mean and std deviation of errors across folds
    avg_errors = np.mean(all_errors, axis=0)
    std_errors = np.std(all_errors, axis=0)
    avg_tests_errors = np.mean(all_tests_errors, axis=0)
    std_tests_errors = np.std(all_tests_errors, axis=0)
    # Plotting average training error with std dev as error bars
    # Define the step
    step = int(epochs/100)
    # Create indices to plot every 100th element
    indices = np.arange(0, len(avg_errors), step)
    plt.figure(figsize=(10, 6))
    plt.errorbar(indices, np.array(avg_errors)[indices], yerr=np.array(std_errors)[indices], fmt='o',
                 label='Average Training Error', color='blue', capsize=5)
    plt.errorbar(indices, np.array(avg_tests_errors)[indices], yerr=np.array(std_tests_errors)[indices], fmt='o',
                 label='Average Test Error', color='red', capsize=5)
    plt.title(f'Average SSE across Epochs with Std Dev ({k_folds}-fold cross-validation)')
    plt.xlabel('Epochs')
    plt.ylabel('SSE')
    plt.xlim(0, len(avg_errors) - 1)
    plt.ylim(0, max(avg_errors + std_errors) * 1.1)
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main(test_size=0.25)  # Specify the number of runs and test size
