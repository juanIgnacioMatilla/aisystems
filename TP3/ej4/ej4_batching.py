from TP3.ej4.mnist_utils import load_preprocessed_mnist, store_preprocessed_mnist, \
    retrieve_preprocessed_mnist, \
    store_model
from TP3.src.model.multilayer_perceptron.vanilla.multi_layer_perceptron import MultiLayerPerceptron


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
    batch_sizes = [
        int(len(x_train)/500),
        int(len(x_train)/1000),
        int(len(x_train)/2000),
        int(len(x_train)/3000)
    ]
    for batch_size in batch_sizes:
        mlp = MultiLayerPerceptron(
            layers_structure=[784, *internal_layers, 10]
        )
        mlp.train(x_train, y_train, epochs=epochs, batch_size=batch_size)  # Adjust the number of epochs as needed
        model_filename = f'trained_models/adam_batch{batch_size}_{epochs}E_784_32_10.pkl'
        store_model(mlp, model_filename)

    mlp = MultiLayerPerceptron(
        layers_structure=[784, *internal_layers, 10]
    )
    mlp.train(x_train, y_train, epochs=epochs)  # Adjust the number of epochs as needed
    model_filename = f'trained_models/adam_online_{epochs}E_784_32_10.pkl'
    store_model(mlp, model_filename)


if __name__ == "__main__":
    main()
