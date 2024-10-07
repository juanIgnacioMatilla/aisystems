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
    values = [i for i in range(100) for _ in range(100)]
    confusion_matrix = np.zeros((10, 10))
    i = 0

    for input in X:
        prediction = mlp.predict(input)
        print(f"Prediction: {np.argmax(prediction)} (Expected: {values[i]}")
        confusion_matrix[values[i]][np.argmax(prediction)] += 1
        i += 1
    # plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
    #
    # sns.heatmap(
    #     confusion_matrix,
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

    # digits_3, _ = load_data("./test_noise_0.3.txt")
    # digits, _ = load_data("./test_noise_0.5.txt")
    digits, _ = load_data("./test_noise_0.7.txt")

    values = [i for i in range(100) for _ in range(100)]

    # mlp = MultiLayerPerceptron(layers_structure=[35, 20, 10], learning_rate=0.1)
    #
    # mlp.train(X, y, epochs=2000)

    # for digits in [digits_3, digits_5, digits_7]:
    #     confusion_matrix = np.zeros((10, 10))
    #     i = 0
    #     for digit in digits:
    #         prediction = mlp.predict(digit)
    #         confusion_matrix[values[i]][np.argmax(prediction)] += 1
    #         i += 1
    #
    #     sns.heatmap(
    #         confusion_matrix,
    #         annot=True,
    #         cmap="YlGnBu",
    #         cbar=True,
    #         square=True,
    #         linewidths=0.5,
    #         fmt="g",
    #     )
    #
    #     # Title and labels
    #     plt.title("Matriz de confusion")
    #     plt.xlabel("Prediccion")
    #     plt.ylabel("Real")
    #
    #     # Show the plot
    #     plt.show()
    #

    metrics = {0: {}, 300: {}, 2100: {}, 4900: {}}
    accuracy_test = []
    accuracy_train = []

    for steps in [0, 300, 2100, 4900]:
        for i in range(10):
            metrics[steps][i] = {
                "recall": [],
                "precision": [],
                "f1": [],
            }
    last_matrix = np.zeros((10, 10))
    errors = []
    for _ in range(5):
        mlp = MultiLayerPerceptron(layers_structure=[35, 20, 10], learning_rate=0.1)
        last_error = []
        loop_metrics = {}
        loop_acc_test = []
        loop_acc_train = []
        for i in range(10):
            loop_metrics[i] = {"recall": [], "precision": [], "f1": [], "accuracy": []}
        for step in range(0, 5000, 50):
            error, _ = mlp.train(X, y, epochs=50)
            last_error = error
            confusion_matrix = np.zeros((10, 10))
            i = 0
            test_count = 0
            train_count = 0
            for input in digits:
                prediction = mlp.predict(input)
                test_count += 1 if values[i] == np.argmax(prediction) else 0
                confusion_matrix[values[i]][np.argmax(prediction)] += 1
                i += 1
            j = 0
            for x in X:
                prediction = mlp.predict(x)
                train_count += 1 if np.argmax(prediction) == np.argmax(y[j]) else 0
                j += 1
            if step in [0, 300, 2100, 4900]:
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
                    metrics[step][i]["recall"].append(recall)
                    metrics[step][i]["precision"].append(precision)
                    metrics[step][i]["f1"].append(f1)
            loop_acc_test.append(test_count / len(digits))
            loop_acc_train.append(train_count / len(X))
            last_error = [last_error[i] for i in range(0, len(last_error), 50)]
            last_matrix = confusion_matrix
        errors.append(last_error)
        accuracy_test.append(loop_acc_test)
        accuracy_train.append(loop_acc_train)

        # plotting
    plt.figure(figsize=(8, 6))  # Adjust the figure size as needed

    sns.heatmap(
        last_matrix,
        annot=True,
        cmap="YlGnBu",
        cbar=True,
        square=True,
        linewidths=0.5,
        fmt="g",
    )

    # Title and labels
    plt.title("Matriz de confusion")
    plt.xlabel("Prediccion")
    plt.ylabel("Real")

    # Show the plot
    plt.show()

    errors = np.array(errors)

    erro_mean = np.mean(errors, axis=0)

    erro_std = np.std(errors, axis=0)
    plt.figure(figsize=(8, 6))
    plt.plot(
        np.arange(0, len(erro_mean) * 50, 50),
        erro_mean,
        label="Error promedio",
        color="red",
    )
    plt.fill_between(
        np.arange(0, len(erro_mean) * 50, 50),
        erro_mean - erro_std,
        erro_mean + erro_std,
        color="red",
        alpha=0.3,
        label="Desviacion estandar",
    )

    plt.title("Error promedio por epoca")
    plt.show()

    accuracy_test = np.array(accuracy_test)
    accuracy_train = np.array(accuracy_train)

    plt.figure(figsize=(8, 6))
    #
    # plt.plot(
    #     np.arange(0, len(accuracy_test[0]) * 50, 50),
    #     np.mean(accuracy_test, axis=0),
    #     label="Test",
    # )
    #
    plt.plot(
        np.arange(0, len(accuracy_train[0]) * 50, 50),
        np.mean(accuracy_train, axis=0),
        label="Sin ruido",
        color="blue",
    )
    plt.fill_between(
        np.arange(0, len(accuracy_train[0]) * 50, 50),
        np.mean(accuracy_train, axis=0) - np.std(accuracy_train, axis=0),
        np.mean(accuracy_train, axis=0) + np.std(accuracy_train, axis=0),
        alpha=0.3,
        label="Desviacion estandar",
        color="blue",
    )
    plt.title("Accuracy promedio por epoca sin ruido")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(
        np.arange(0, len(accuracy_test[0]) * 50, 50),
        np.mean(accuracy_test, axis=0),
        label="Con ruido",
        color="orange",
    )

    plt.fill_between(
        np.arange(0, len(accuracy_test[0]) * 50, 50),
        np.mean(accuracy_test, axis=0) - np.std(accuracy_test, axis=0),
        np.mean(accuracy_test, axis=0) + np.std(accuracy_test, axis=0),
        alpha=0.3,
        label="Desviacion estandar",
        color="orange",
    )
    plt.title("Accuracy promedio por epoca con ruido")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()
    #

    plt.figure(figsize=(8, 6))

    plt.plot(
        np.arange(0, len(accuracy_test[0]) * 50, 50),
        np.mean(accuracy_test, axis=0),
        label="Con ruido",
        color="orange",
    )
    plt.fill_between(
        np.arange(0, len(accuracy_test[0]) * 50, 50),
        np.mean(accuracy_test, axis=0) - np.std(accuracy_test, axis=0),
        np.mean(accuracy_test, axis=0) + np.std(accuracy_test, axis=0),
        alpha=0.3,
        label="Desviacion estandar",
        color="orange",
    )

    plt.plot(
        np.arange(0, len(accuracy_train[0]) * 50, 50),
        np.mean(accuracy_train, axis=0),
        label="Sin ruido",
        color="blue",
    )
    plt.fill_between(
        np.arange(0, len(accuracy_train[0]) * 50, 50),
        np.mean(accuracy_train, axis=0) - np.std(accuracy_train, axis=0),
        np.mean(accuracy_train, axis=0) + np.std(accuracy_train, axis=0),
        alpha=0.3,
        label="Desviacion estandar",
        color="blue",
    )
    plt.title("Accuracy promedio por epoca ")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()

    metric_names = ["precision", "recall", "f1"]
    print(metrics)
    for step in [0, 300, 2100, 4900]:
        for metric in metric_names:
            means = []
            stds = []
            for digit in range(10):
                means.append(np.mean(metrics[step][digit][metric]))
                stds.append(np.std(metrics[step][digit][metric]))
            plt.figure(figsize=(8, 6))
            plt.bar(range(10), means, yerr=stds, capsize=5)
            plt.xticks(range(10), [str(i) for i in range(10)])
            plt.title(f"{metric} for each digit in epoch {step}")
            plt.xlabel("Digit")
            plt.ylabel(metric)
            plt.ylim(0, 1.3)
            plt.show()


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
