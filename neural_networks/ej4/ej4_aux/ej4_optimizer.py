
from neural_networks.ej4.mnist_utils import load_preprocessed_mnist, store_preprocessed_mnist, \
    retrieve_preprocessed_mnist, \
    store_model
from neural_networks.src.model.multilayer_perceptron.momentum.momentum_multi_layer_perceptron import MomentumMultiLayerPerceptron
from neural_networks.src.model.multilayer_perceptron.vanilla.multi_layer_perceptron import MultiLayerPerceptron


def main():
    # Load the preprocessed MNIST dataset
    (x_train, y_train), (x_test, y_test) = load_preprocessed_mnist()

    #store the preprocessed MNIST dataset
    store_preprocessed_mnist(x_train, y_train, x_test, y_test)
    # Load the stored preprocessed MNIST dataset
    (x_train, y_train), (x_test, y_test) = retrieve_preprocessed_mnist()
    epochs = 10
    internal_layers = [32]
    print("Vanilla")
    vanilla_mlp = MultiLayerPerceptron(
        layers_structure=[784, *internal_layers, 10]
    )
    vanilla_mlp.train(x_train, y_train, epochs=epochs)  # Adjust the number of epochs as needed
    model_filename = f'trained_models/adam_{epochs}E_784_32_10.pkl'
    store_model(vanilla_mlp, model_filename)

    print("Momentum")
    momentum_mlp = MomentumMultiLayerPerceptron(
        layers_structure=[784, *internal_layers, 10]
    )
    momentum_mlp.train(x_train, y_train, epochs=epochs)  # Adjust the number of epochs as needed
    model_filename = f'trained_models/adam_{epochs}E_784_32_10.pkl'
    store_model(momentum_mlp, model_filename)

    print("Adam")
    adam_mlp = MomentumMultiLayerPerceptron(
        layers_structure=[784, *internal_layers, 10]
    )
    adam_mlp.train(x_train, y_train, epochs=epochs)  # Adjust the number of epochs as needed
    model_filename = f'trained_models/adam_{epochs}E_784_32_10.pkl'
    store_model(adam_mlp, model_filename)

if __name__ == "__main__":
    main()
