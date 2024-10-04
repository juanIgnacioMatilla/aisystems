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


# Helper function for manual one-hot encoding
def one_hot_encode(labels, num_classes):
    one_hot_labels = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        one_hot_labels[i][label] = 1
    return one_hot_labels


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
    labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    # Manually one-hot encode the labels
    labels_one_hot = one_hot_encode(labels, 10)
    return np.array(digits), labels_one_hot


def load_config(file_path="./config.json"):
    with open(file_path, "r") as file:
        config = json.load(file)
    return config


def train(X, y, config):
    learning_rate = config.get("learning_rate", 0.1)
    epochs = config.get("epochs", 5000)
    structure = [35] + config.get("structure", [10, 5]) + [10]
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


def test(mlp, X):
    values = [i for i in range(10) for _ in range(10)]
    confusion_matrix = np.zeros((10, 10))
    i = 0
    for input in X:
        prediction = mlp.predict(input)
        print(f"Prediction: {np.argmax(prediction)} (Expected: {values[i]}")
        confusion_matrix[values[i]][np.argmax(prediction)] += 1
        i += 1
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
    for i in range(10):
        tp = confusion_matrix[i][i]
        tn = sum([confusion_matrix[j][j] for j in range(10) if j != i])
        fp = sum([confusion_matrix[j][i] for j in range(10) if j != i])
        fn = sum([confusion_matrix[i][j] for j in range(10) if j != i])
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
        plt.bar(range(10), [metrics[i][metric] for i in range(10)])
        plt.xticks(range(10), [str(i) for i in range(10)])
        plt.title(f"{metric} for each digit")
        plt.xlabel("Digit")
        plt.ylabel(metric)
        plt.show()


def main():
    configs = load_config()

    X, y = load_data("../inputs/TP3-ej3-digitos.txt")

    for config in configs:
        print(config)
        mlp, errors = train(X, y, config)
        digits, _ = load_data("./test_noise_0.2.txt")
        test(mlp, digits)

    # # Assuming the Layer and MultiLayerPerceptron classes are defined
    # # Load the data from the file
    # # Define the MLP structure
    # # Input size is 35 (flattened 7x5 digits), one hidden layer with 10 neurons, and 1 output
    # # mlp = MultiLayerPerceptron(layers_structure=[35, 20, 10], learning_rate=0.1)
    # vanilla_mlp = MultiLayerPerceptron(layers_structure=[35, 20, 10], learning_rate=0.1)
    #
    # # Train the MLP
    # errors = vanilla_mlp.train(X, y, epochs=5000)
    # print("Vanilla: ")
    # for i, x in enumerate(X):
    #     predictions = vanilla_mlp.predict(x)
    #     predicted_digit = np.argmax(
    #         predictions
    #     )  # Get the index of the highest probability (predicted digit)
    #     print(
    #         f"Prediction for digit {i}: {predicted_digit} (Expected: {np.argmax(y[i])})"
    #     )
    #     print(predictions)
    #     print()
    #
    # for i, error in enumerate(errors):
    #     if i % 1000 == 0:
    #         print("error for epoch ", i, ": ", error)
    #
    # # Define the MLP structure
    # # Input size is 35 (flattened 7x5 digits), hidden layer with 100 neurons, output layer with 10 neurons
    # adam_mlp = AdamMultiLayerPerceptron(layers_structure=[35, 20, 10])
    #
    # # Train the MLP
    # errors, accuracies = adam_mlp.train(X, y, epochs=5000)
    # print("Adam: ")
    # # Make predictions and display results
    # for i, x in enumerate(X):
    #     predictions = adam_mlp.predict(x)
    #     predicted_digit = np.argmax(
    #         predictions
    #     )  # Get the index of the highest probability (predicted digit)
    #     print(
    #         f"Prediction for digit {i}: {predicted_digit} (Expected: {np.argmax(y[i])})"
    #     )
    #     print(predictions)
    #     print()
    #
    # # Display errors for some epochs
    # for i, error in enumerate(errors):
    #     if i % 1000 == 0:
    #         print(f"Error for epoch {i}: {error}")
    #


if __name__ == "__main__":
    main()
