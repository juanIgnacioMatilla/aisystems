import time

import numpy as np

from TP3.ej4.mnist_utils import load_preprocessed_mnist, load_preprocessed_mnist_sample, store_preprocessed_mnist, \
    retrieve_preprocessed_mnist, \
    store_model, load_model, accuracy_test_set, graph_accuracy_v_epochs, error_test_set
from TP3.src.model.multilayer_perceptron.adam.adam_multi_layer_perceptron import AdamMultiLayerPerceptron
from TP3.src.model.multilayer_perceptron.vanilla.multi_layer_perceptron import MultiLayerPerceptron


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




    epochs = 100
    internal_layers = [10]

    adam_mlp = MultiLayerPerceptron(
        layers_structure=[784, *internal_layers, 10]
    )

    train_set_accuracies = []
    train_set_errors = []
    test_set_accuracies = []
    test_set_errors = []
    # (x_train_subset, y_train_subset), (x_test, y_test) = load_preprocessed_mnist_sample(sample_size=500)

    initial_time = time.time()

    for i in range(epochs):
        adam_mlp.train(x_train, y_train, epochs=1)
        train_set_accuracies.append(adam_mlp.accuracies_by_epoch[-1])
        train_set_errors.append(adam_mlp.errors_by_epoch[-1])
        test_set_accuracies.append(accuracy_test_set(adam_mlp, x_test, y_test))
        test_set_errors.append(error_test_set(adam_mlp, x_test, y_test))

    print(f"Training time: {time.time() - initial_time}")

    # graph the results in one plot
    import matplotlib.pyplot as plt
    import numpy as np

    plt.plot(np.arange(1, epochs + 1), train_set_accuracies, label='Train set')
    plt.plot(np.arange(1, epochs + 1), test_set_accuracies, label='Test set')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train and Test set accuracies')
    plt.legend()
    plt.show()

    plt.plot(np.arange(1, epochs + 1), train_set_errors, label='Train set')
    plt.plot(np.arange(1, epochs + 1), test_set_errors, label='Test set')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Train and Test set errors')
    plt.legend()
    plt.show()

    # #artificial underfitting
    # #large training set and small number of epochs
    # epochs = 10
    # internal_layers = [10]
    #
    # vanilla_mlp = MultiLayerPerceptron(
    #     layers_structure=[784, *internal_layers, 10]
    # )
    #
    # vanilla_mlp.train(x_train, y_train, epochs=epochs)  # Adjust the number of epochs as needed
    # model_filename = f'trained_models/UNDERFITTED_VANILLA_{epochs}E_784_10_10.pkl'
    # store_model(vanilla_mlp, model_filename)


if __name__ == "__main__":
    main()
