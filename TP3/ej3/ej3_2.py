import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from TP3.src.model.multilayer_perceptron.adam.adam_multi_layer_perceptron import AdamMultiLayerPerceptron
from TP3.src.model.multilayer_perceptron.momentum.momentum_multi_layer_perceptron import MomentumMultiLayerPerceptron
from TP3.src.model.multilayer_perceptron.vanilla.multi_layer_perceptron import MultiLayerPerceptron

# Helper function to read and preprocess the data
def load_data(file_path):
    digits = []
    with open(file_path, 'r') as file:
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
    structure = [35] + config.get("structure", [10, 5]) + [2]
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
    errors, accuracy = mlp[optimizer].train(X, y, epochs=epochs)
    return errors, accuracy, mlp[optimizer]


def train_recall(X, y, config):
    learning_rate = config.get("learning_rate", 0.1)
    epochs = config.get("epochs", 5000)
    structure = [35] + config.get("structure", [10, 5]) + [2]
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
    errors, accuracy = mlp[optimizer].train(X, y, epochs=epochs)

    recall_values = []
    for epoch in range(epochs):
        confusion_matrix = np.zeros((10, 10))
        for i, (input, target) in enumerate(zip(X, y)):
            prediction = mlp[optimizer].predict(input)
            predicted_label = np.argmax(prediction)
            true_label = np.argmax(target)
            confusion_matrix[true_label][predicted_label] += 1

        tp = confusion_matrix[epoch % 10][epoch % 10]
        fn = sum([confusion_matrix[epoch % 10][j] for j in range(10) if j != epoch % 10])
        recall = tp / (tp + fn) if tp + fn != 0 else 0
        recall_values.append(recall)

    return errors, accuracy, mlp[optimizer], recall_values


def main():
    configs = load_config()

    # Load the data from the file
    X, y = load_data('../inputs/TP3-ej3-digitos.txt')

    for config in configs:
        print(config)

        # Initialize lists to store recall for each run
        all_accuracies = []

        for _ in range(config.get("k")):
            errors, accuracy, mlp = train(X, y, config)
            partial_accuracies = [accuracy[i] for i in range(0, len(accuracy), 75)]
            all_accuracies.append(partial_accuracies)

        # Convert to a NumPy array for easier manipulation
        all_accuracies = np.array(all_accuracies)

        # Calculate mean and standard deviation of accuracies across runs
        accuracy_mean = np.mean(all_accuracies, axis=0)
        accuracy_std = np.std(all_accuracies, axis=0)

        # Plot mean accuracy with y-error bars
        epochs = np.arange(0, len(accuracy_mean) * 75, 75)  # Epoch indices (sampled every 5 epochs)
        plt.figure(figsize=(12, 6))
        plt.errorbar(epochs, accuracy_mean, yerr=accuracy_std, fmt='o', capsize=5, label='Average Accuracy',
                     color='blue')

        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main()