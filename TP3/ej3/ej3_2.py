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
    errors, _ = mlp[optimizer].train(X, y, epochs=epochs)
    return mlp[optimizer], errors


def test(mlp, X, y):
    confusion_matrix = np.zeros((2, 2))
    for i, x in enumerate(X):
        predictions = mlp.predict(x)
        print(
            "Prediction for digit",
            i,
            ": ",
            "even" if np.round(predictions) == 0 else "odd",
        )
        print()
        confusion_matrix[y[i]][int(np.round(predictions))] += 1

    plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
    sns.heatmap(
        confusion_matrix,
        annot=True,
        cmap="YlGnBu",
        cbar=True,
        square=True,
        linewidths=0.5,
    )
    # Title and labels
    plt.title("Matriz de confusion")
    plt.xlabel("Prediccion")
    plt.ylabel("Real")
    # Show the plot
    plt.show()
    metrics = {}
    for i in range(2):

        tp = confusion_matrix[i][i]
        tn = sum([confusion_matrix[j][j] for j in range(2) if j != i])
        fp = sum([confusion_matrix[j][i] for j in range(2) if j != i])
        fn = sum([confusion_matrix[i][j] for j in range(2) if j != i])
        precision = tp / (tp + fp) if tp + fp != 0 else 0

        recall = tp / (tp + fn) if tp + fn != 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if precision + recall != 0
            else 0
        )
        accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn != 0 else 0
        tp_rate = tp / (tp + fn) if tp + fn != 0 else 0
        fp_rate = fp / (fp + tn) if fp + tn != 0 else 0
        metrics[i] = {  # Save metrics for each digit
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Accuracy": accuracy,
            "TP Rate": tp_rate,
            "FP Rate": fp_rate,
        }

    # plot each metric
    metrics_names = ["Precision", "Recall", "F1", "Accuracy", "TP Rate", "FP Rate"]
    for metric in metrics_names:
        plt.figure(figsize=(8, 6))
        plt.bar(range(2), [metrics[i][metric] for i in range(2)])
        plt.xticks(range(2), [str(i) for i in range(2)])
        plt.title(f"{metric} for each digit")
        plt.xlabel("Digit")
        plt.ylabel(metric)
        plt.show()


def main():

    configs = load_config()
    X, y = load_data("../inputs/TP3-ej3-digitos.txt")
    for config in configs:
        mlp, _ = train(X, y, config)
        test(mlp, X, y)


if __name__ == "__main__":
    main()

