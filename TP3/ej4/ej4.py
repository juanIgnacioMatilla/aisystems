import numpy as np

from TP3.ej4.mnist_utils import load_preprocessed_mnist, load_preprocessed_mnist_sample, store_preprocessed_mnist, \
    retrieve_preprocessed_mnist, \
    store_model, load_model
from TP3.src.model.multilayer_perceptron.adam.adam_multi_layer_perceptron import AdamMultiLayerPerceptron

# EPOCHS = 7


def main():
    # ReLU
    relu = lambda x: np.maximum(0, x)
    relu_derivative = lambda x: np.where(x <= 0, 0, 1)

    # Leaky ReLU
    leaky_relu = lambda x: np.where(x > 0, x, 0.01 * x)
    leaky_relu_derivative = lambda x: np.where(x > 0, 1, 0.01)

    # Tanh
    tanh = lambda x: np.tanh(x)
    tanh_derivative = lambda x: 1 - np.tanh(x) ** 2

    # ELU (Exponential Linear Unit)
    elu = lambda x: np.where(x > 0, x, np.exp(x) - 1)
    elu_derivative = lambda x: np.where(x > 0, 1, np.exp(x))

    # # Load the preprocessed MNIST dataset
    # (x_train, y_train), (x_test, y_test) = load_preprocessed_mnist()
    #
    # #store the preprocessed MNIST dataset
    # store_preprocessed_mnist(x_train, y_train, x_test, y_test)
    #
    # # Load a subset of the preprocessed MNIST dataset
    # (x_train_subset, y_train_subset),(x_test_subset, y_test_subset) = load_preprocessed_mnist_sample(sample_size=1000)

    # Load the stored preprocessed MNIST dataset
    (x_train, y_train), (x_test, y_test) = retrieve_preprocessed_mnist()

    # Define the MLP structure
    # Input size is 784 (flattened 28x28 images), hidden layer with 64 neurons, output layer with 10 neurons
    # adam_mlp = AdamMultiLayerPerceptron(layers_structure=[784, 64, 10])


    # # Train the MLP with Adam optimizer
    # adam_mlp.train(x_train, y_train, epochs=EPOCHS)  # Adjust the number of epochs as needed
    # # adam_mlp.train(x_train_subset, y_train_subset, epochs=2)  # Adjust the number of epochs as needed
    # print("\nTraining with Adam optimizer completed.\n")
    #
    # model_filename = f'trained_models/ADAM_{EPOCHS}E.pkl'
    # store_model(adam_mlp, model_filename)

    epochs = 10
    internal_layers = [256, 128, 64]

    adam_mlp = AdamMultiLayerPerceptron(
        layers_structure=[784, *internal_layers, 10]
    )
    adam_mlp.train(x_train, y_train, epochs=epochs)  # Adjust the number of epochs as needed
    model_filename = f'trained_models/ADAM_{epochs}E_784_256_128_64_10.pkl'
    store_model(adam_mlp, model_filename)

if __name__ == "__main__":
    main()
