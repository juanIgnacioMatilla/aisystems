import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from TP3.src.model.multilayer_perceptron.adam.adam_multi_layer_perceptron import (
    AdamMultiLayerPerceptron,
)
from TP3.src.model.multilayer_perceptron.momentum.momentum_multi_layer_perceptron import (
    MomentumMultiLayerPerceptron,
)
from TP3.src.model.multilayer_perceptron.vanilla.multi_layer_perceptron import (
    MultiLayerPerceptron,
)


# Helper function to read and preprocess the data
def load_data(file_path):
    digits = []
    with open(file_path, "r") as file:
        lines = file.readlines()
        digit_data = []
        for i, line in enumerate(lines):
            line = line.strip()
            if line:
                digit_data.append([int(x) for x in line.split()])
            if (i + 1) % 7 == 0:  # Each digit is represented by 7 lines
                digits.append(np.array(digit_data).flatten())
                digit_data = []

    # The targets: 0 for even, 1 for odd (digits 0-9)
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    return np.array(digits), labels


def calculate_accuracy(predictions, targets):
    return np.mean(np.round(predictions) == targets)


def load_config(file_path="./config.json"):
    with open(file_path, "r") as file:
        config = json.load(file)
    return config


def train(X, y, config):
    learning_rate = config.get("learning_rate", 0.1)
    epochs = config.get("epochs", 5000)
    structure = [35] + config.get("structure", [10, 5]) + [1]
    alpha = config.get("alpha", 0.9)
    optimizer = config.get("optimizer", "vanilla")
    mlp = {
        "vanilla": MultiLayerPerceptron(
            learning_rate=learning_rate, layers_structure=structure
        ),
        "adam": AdamMultiLayerPerceptron(
            learning_rate=learning_rate, layers_structure=structure
        ),
        "momentum": MomentumMultiLayerPerceptron(
            learning_rate=learning_rate, layers_structure=structure, alpha=alpha
        ),
    }
    errors, accuracy, matrices_by_epoch = mlp[optimizer].train(X, y, epochs=epochs)
    return (
        errors,
        accuracy,
        mlp[optimizer],
        matrices_by_epoch,
    )


def main():
    configs = load_config()

    # Load the data from the file
    X, y = load_data("../inputs/TP3-ej3-digitos.txt")

    even_recalls = []
    odd_recalls = []
    even_precisions = []
    odd_precisions = []
    even_f1s = []
    odd_f1s = []
    accuracies = []
    last = None
    errors = []
    # for _ in range(5):
    #     (
    #         even_recall,
    #         odd_recall,
    #         even_precision,
    #         odd_precision,
    #         even_f1,
    #         odd_f1,
    #         accuracy,
    #         last_matrix,
    #         error,
    #     ) = run(X, y)
    #     even_recalls.append(even_recall)
    #     odd_recalls.append(odd_recall)
    #     even_precisions.append(even_precision)
    #     odd_precisions.append(odd_precision)
    #     even_f1s.append(even_f1)
    #     odd_f1s.append(odd_f1)
    #     accuracies.append(accuracy)
    #     last = last_matrix
    #     error = [error[i] for i in range(0, 1000, 10)]
    #     errors.append(error)
    # even_recalls = np.array(even_recalls)
    # odd_recalls = np.array(odd_recalls)
    # even_recall_mean = np.mean(even_recalls, axis=0)
    # even_recall_std = np.std(even_recalls, axis=0)
    # odd_recall_mean = np.mean(odd_recalls, axis=0)
    # odd_recall_std = np.std(odd_recalls, axis=0)
    # plot_figure(
    #     even_recall_mean, odd_recall_mean, even_recall_std, odd_recall_std, "Recall"
    # )
    #
    # even_precisions = np.array(even_precisions)
    # odd_precisions = np.array(odd_precisions)
    # even_precision_mean = np.mean(even_precisions, axis=0)
    # even_precision_std = np.std(even_precisions, axis=0)
    # odd_precision_mean = np.mean(odd_precisions, axis=0)
    # odd_precision_std = np.std(odd_precisions, axis=0)
    # plot_figure(
    #     even_precision_mean,
    #     odd_precision_mean,
    #     even_precision_std,
    #     odd_precision_std,
    #     "Precision",
    # )
    #
    # even_f1s = np.array(even_f1s)
    # odd_f1s = np.array(odd_f1s)
    # even_f1_mean = np.mean(even_f1s, axis=0)
    # even_f1_std = np.std(even_f1s, axis=0)
    # odd_f1_mean = np.mean(odd_f1s, axis=0)
    # odd_f1_std = np.std(odd_f1s, axis=0)
    # plot_figure(even_f1_mean, odd_f1_mean, even_f1_std, odd_f1_std, "F1 Score")
    # #
    # # errors = np.array(errors)
    # # error_mean = np.mean(errors, axis=0)
    # # error_std = np.std(errors, axis=0)
    # #
    # # plt.errorbar(
    # #     np.arange(0, len(error_mean) * 15, 15),
    # #     error_mean,
    # #     yerr=error_std,
    # #     fmt="o",
    # #     capsize=5,
    # #     label=f"Error",
    # #     color="blue",
    # # )
    # #
    # # plt.xlabel("Epochs")
    # # plt.ylabel("Error")
    # # plt.title("Error vs Epochs")
    # # plt.legend()
    # # plt.grid(True)
    # # plt.show()
    # #
    # # accuracies = np.array(accuracies)
    # # accuracy_mean = np.mean(accuracies, axis=0)
    # # accuracy_std = np.std(accuracies, axis=0)
    # # plt.errorbar(
    # #     np.arange(0, len(accuracy_mean) * 15, 15),
    # #     accuracy_mean,
    # #     yerr=accuracy_std,
    # #     fmt="o",
    # #     capsize=5,
    # #     label=f"Accuracy",
    # #     color="blue",
    # # )
    # #
    # # plt.xlabel("Epochs")
    # # plt.ylabel("Accuracy")
    # # plt.title("Accuracy vs Epochs")
    # # plt.legend()
    # # plt.grid(True)
    # # plt.show()
    # #
    #
    # # Convert to arrays
    # errors = np.array(errors)
    # error_mean = np.mean(errors, axis=0)
    # error_std = np.std(errors, axis=0)
    #
    # # Set up epochs
    # epochs = np.arange(0, len(error_mean) * 10, 10)
    #
    # # Plot the error mean with shaded region for error standard deviation
    # plt.plot(epochs, error_mean, label=f"Error", color="blue")
    # plt.fill_between(
    #     epochs,
    #     error_mean - error_std,
    #     error_mean + error_std,
    #     color="blue",
    #     alpha=0.3,  # Transparency for the shadow
    #     label="Error Range",
    # )
    #
    # # Labels, title, and grid
    # plt.xlabel("Epochs")
    # plt.ylabel("Error")
    # plt.title("Error vs Epochs")
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    #
    # # Convert accuracies to arrays
    # accuracies = np.array(accuracies)
    # accuracy_mean = np.mean(accuracies, axis=0)
    # accuracy_std = np.std(accuracies, axis=0)
    #
    # # Set up epochs
    # epochs = np.arange(0, len(accuracy_mean) * 10, 10)
    #
    # # Plot the accuracy mean with shaded region for accuracy standard deviation
    # plt.plot(epochs, accuracy_mean, label=f"Accuracy", color="blue")
    # plt.fill_between(
    #     epochs,
    #     accuracy_mean - accuracy_std,
    #     accuracy_mean + accuracy_std,
    #     color="blue",
    #     alpha=0.3,  # Transparency for the shadow
    #     label="Accuracy Range",
    # )
    #
    # # Labels, title, and grid
    # plt.xlabel("Epochs")
    # plt.ylabel("Accuracy")
    # plt.title("Accuracy vs Epochs")
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    # plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
    #
    # sns.heatmap(
    #     last,
    #     annot=True,
    #     cmap="YlGnBu",
    #     cbar=True,
    #     square=True,
    #     linewidths=0.5,
    # )
    #
    # # Title and labels
    # plt.title("Matriz de confusion")
    # plt.xlabel("Prediccion")
    # plt.ylabel("Real")
    #
    # # Show the plot
    # plt.show()
    #
    l_rates = [0.1, 0.01, 0.001]
    err_dict = {}

    for rate in l_rates:
        err_dict[rate] = []
        for _ in range(5):
            mlp = MultiLayerPerceptron(
                layers_structure=[35, 10, 5, 1], learning_rate=0.1
            )
            errors, _ = mlp.train(X, y, epochs=100)
            err_dict[rate].append(errors)

    plt.figure(figsize=(12, 6))
    for rate, color in zip(err_dict.keys(), ["green", "blue", "orange"]):
        errors = np.array(err_dict[rate])
        error_mean = np.mean(errors, axis=0)
        error_std = np.std(errors, axis=0)
        epochs = np.arange(0, len(error_mean))
        plt.plot(epochs, error_mean, label=f"Learning Rate {rate}", color=color)
        plt.fill_between(
            epochs,
            error_mean - error_std,
            error_mean + error_std,
            color=color,
            alpha=0.3,
        )
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.title("Error vs Epochs")
    plt.legend()
    plt.show()


def plot_figure(even, odd, even_std, odd_std, title):
    plt.figure(figsize=(12, 6))
    plt.errorbar(
        np.arange(0, len(even) * 10, 10),
        even,
        yerr=even_std,
        fmt="o",
        capsize=5,
        label=f"Even {title}",
        color="blue",
    )
    plt.errorbar(
        np.arange(0, len(odd) * 10, 10),
        odd,
        yerr=odd_std,
        fmt="o",
        capsize=5,
        label=f"Odd {title}",
        color="red",
    )
    plt.xlabel("Epochs")
    plt.ylabel(title)
    plt.title(f"{title} vs Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()


def run(X, y):
    mlp = MultiLayerPerceptron(layers_structure=[35, 10, 5, 1], learning_rate=0.1)

    count = 0
    step = 10
    matrices = []
    matrix = np.zeros((2, 2))
    errors = None
    for i, x in enumerate(X):
        predictions = mlp.predict(x)

        if np.round(predictions) == 0:
            if y[i] == 0:
                matrix[0, 0] += 1
            else:
                matrix[0, 1] += 1
        else:
            if y[i] == 1:
                matrix[1, 1] += 1
            else:
                matrix[1, 0] += 1
    matrices.append(matrix)

    while count < 1000:
        error, _ = mlp.train(X, y, epochs=step)
        matrix = np.zeros((2, 2))
        for i, x in enumerate(X):
            predictions = mlp.predict(x)

            if np.round(predictions) == 0:
                if y[i] == 0:
                    matrix[0, 0] += 1
                else:
                    matrix[0, 1] += 1
            else:
                if y[i] == 1:
                    matrix[1, 1] += 1
                else:
                    matrix[1, 0] += 1
        matrices.append(matrix)
        count += step
        errors = error

    even_recall = []
    odd_recall = []
    even_precision = []
    odd_precision = []
    even_f1 = []
    odd_f1 = []
    accuracy = []

    for i, matrix in enumerate(matrices):
        tn = matrix[0, 0]
        tp = matrix[1, 1]
        fn = matrix[0, 1]
        fp = matrix[1, 0]
        even_recall.append(tp / (tp + fn) if tp + fn != 0 else 0)
        odd_recall.append(tn / (tn + fp) if tn + fp != 0 else 0)
        even_precision.append(tp / (tp + fp) if tp + fp != 0 else 0)
        odd_precision.append(tn / (tn + fn) if tn + fn != 0 else 0)
        even_f1.append(
            2
            * (even_precision[i] * even_recall[i])
            / (even_precision[i] + even_recall[i])
            if even_precision[i] + even_recall[i] != 0
            else 0
        )
        odd_f1.append(
            2 * (odd_precision[i] * odd_recall[i]) / (odd_precision[i] + odd_recall[i])
            if odd_precision[i] + odd_recall[i] != 0
            else 0
        )
        accuracy.append((tp + tn) / (tp + tn + fp + fn))
    return (
        even_recall,
        odd_recall,
        even_precision,
        odd_precision,
        even_f1,
        odd_f1,
        accuracy,
        matrices[-1],
        errors,
    )


if __name__ == "__main__":
    main()
