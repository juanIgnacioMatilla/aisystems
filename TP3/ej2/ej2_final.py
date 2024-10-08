import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from TP3.src.model.simple_perceptron.simple_nonlineal_perceptron import SimpleNonLinearPerceptron


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


def k_fold_split(X, k_folds=5):
    """
    Divide los datos en k subconjuntos.
    Retorna una lista de índices de entrenamiento y prueba para cada fold.
    """
    indices = np.arange(len(X))
    np.random.shuffle(indices)  # Mezcla los índices para asegurar aleatoriedad
    fold_size = len(X) // k_folds
    folds = []

    for k in range(k_folds):
        test_indices = indices[k * fold_size:(k + 1) * fold_size]
        train_indices = np.setdiff1d(indices, test_indices)
        folds.append((train_indices, test_indices))

    return folds


def main(runs=5, k_folds=5):  # Add k_folds parameter
    # Load data, specify that the first row is the header
    data = pd.read_csv("../inputs/TP3-ej2-conjunto.csv", header=0)
    # Extract features and target variable
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].astype(float).values  # Convert target to float
    y_min, y_max = y.min(), y.max()
    y = normalize_data(y, y_min=y_min, y_max=y_max)
    # Generate k-fold splits
    folds = k_fold_split(X, k_folds=k_folds)
    all_errors = []  # List to store errors from each fold
    all_tests_errors = []

    for fold, (train_index, test_index) in enumerate(folds):
        print(f"Starting fold {fold + 1} / {k_folds}")

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        fold_errors = []  # Store errors for each run within the fold
        fold_test_errors = []

        for run in range(runs):
            # Train the nonlinear perceptron
            nonlinear_perceptron = SimpleNonLinearPerceptron(
                learning_rate=0.001,
                activation_function=tanh,
                activation_function_derivate=tanh_deriv
            )

            initial_weights = np.random.uniform(-0.1, 0.1, size=(len(X_train[0]) + 1))

            nonlinear_neuron, errors, neurons = nonlinear_perceptron.train(
                X_train, y_train, y_max, y_min, initial_weights, epochs=2000
            )
            tests_errors_by_epoch = []
            for epoch in range(2000):
                epoch_error = 0
                for inputs, target in zip(X_test, y_test):
                    prediction = neurons[epoch].predict(inputs)
                    epoch_error += 0.5 * (denormalize_data(target,y_min, y_max) - denormalize_data(prediction,y_min, y_max)) ** 2
                tests_errors_by_epoch.append(epoch_error)
            fold_test_errors.append(tests_errors_by_epoch)
            fold_errors.append(errors)  # Store errors for this run

        all_errors.append(np.mean(fold_errors, axis=0))  # Average errors for the fold
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
    step = 20
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
    main(runs=5, k_folds=5)  # Specify the number of runs and k-folds
