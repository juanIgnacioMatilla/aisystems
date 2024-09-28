import numpy as np

from TP3.src.model.multilayer_perceptron.mult_layer_perceptron import MultiLayerPerceptron

# Helper function for manual one-hot encoding
def one_hot_encode(labels, num_classes):
    one_hot_labels = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        one_hot_labels[i][label] = 1
    return one_hot_labels

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
    labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    # Manually one-hot encode the labels
    labels_one_hot = one_hot_encode(labels, 10)
    return np.array(digits), labels_one_hot


def main():
    # Assuming the Layer and MultiLayerPerceptron classes are defined

    # Load the data from the file
    X, y = load_data('../inputs/TP3-ej3-digitos.txt')
    # Define the MLP structure
    # Input size is 35 (flattened 7x5 digits), one hidden layer with 10 neurons, and 1 output
    # mlp = MultiLayerPerceptron(layers_structure=[35, 20, 10], learning_rate=0.1)
    mlp = MultiLayerPerceptron(layers_structure=[35, 100, 10], learning_rate=0.1)

    # Train the MLP
    errors = mlp.train(X, y, epochs=5000)

    for i, x in enumerate(X):
        predictions = mlp.predict(x)
        predicted_digit = np.argmax(predictions)  # Get the index of the highest probability (predicted digit)
        print(f"Prediction for digit {i}: {predicted_digit} (Expected: {np.argmax(y[i])})")
        print(predictions)
        print()

    for i, error in enumerate(errors):
        if i % 1000 == 0:
            print("error for epoch ",i,": ",error)


if __name__ == "__main__":
    main()
