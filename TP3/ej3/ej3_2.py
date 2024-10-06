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
    errors, accuracy, recalls = mlp[optimizer].train(X, y, epochs=epochs)
    return errors, accuracy, recalls, mlp[optimizer]


def main():
    configs = load_config()

    # Load the data from the file
    X, y = load_data('../inputs/TP3-ej3-digitos.txt')

    for config in configs:
        print(config)

        # Initialize lists to store recall for each run
        all_recalls = []

        for _ in range(config.get("k")):
            errors, accuracy, recalls, mlp = train(X, y, config)
            sampled_recalls = [recalls[i] for i in range(0, len(recalls), 75)]
            all_recalls.append(sampled_recalls)

        # Convert to a NumPy array for easier manipulation
        all_recalls = np.array(all_recalls)

        # Calculate mean and standard deviation of recalls across runs
        recall_mean = np.mean(all_recalls, axis=0)
        recall_std = np.std(all_recalls, axis=0)

        # Plot mean recall with y-error bars
        epochs = np.arange(0, len(recall_mean) * 5, 5)  # Epoch indices (sampled every 5 epochs)
        plt.figure(figsize=(12, 6))
        plt.errorbar(epochs, recall_mean, yerr=recall_std, fmt='o', capsize=5, label='Average Recall', color='blue')

        plt.xlabel('Epochs')
        plt.ylabel('Recall')
        plt.title('Recall vs Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main()