import numpy as np

from TP3.src.model.multilayer_perceptron.mult_layer_perceptron import MultiLayerPerceptron


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


def main():
    # Assuming the Layer and MultiLayerPerceptron classes are defined

    # Load the data from the file
    X, y = load_data('./inputs/TP3-ej3-digitos.txt')
    # Define the MLP structure
    # Input size is 35 (flattened 7x5 digits), one hidden layer with 10 neurons, and 1 output
    mlp = MultiLayerPerceptron(layers_structure=[35, 10, 1], learning_rate=0.1)

    # Train the MLP
    errors = mlp.train(X, y, epochs=5000)

    # Test the MLP
    for i, x in enumerate(X):
        predictions = mlp.predict(x)
        print("prediction for ", i, ": ", "even" if np.round(predictions) == 0 else "odd")  # Round the output to 0 or 1
        print()
    for i,error in enumerate(errors):
        if i % 1000 == 0:
            print("error for epoch ",i,": ",error)

if __name__ == "__main__":
    main()
