from TP3.ej4.mnist_utils import print_training_errors, print_training_accuracies, \
    load_model, print_n_predictions, retrieve_preprocessed_mnist


def main():
    # Load the preprocessed MNIST dataset
    (_, _), (x_test, y_test) = retrieve_preprocessed_mnist()

    # Load the trained model from a file
    model_filename = 'trained_models/trained_mlp_ADAM_4E.pkl'
    loaded_mlp = load_model(model_filename)

    print_n_predictions(loaded_mlp, x_test, y_test, n=10)
    print_training_errors(loaded_mlp)
    print_training_accuracies(loaded_mlp)

    # retrain the model
    # errors, accuracies = loaded_mlp.train(x_train, y_train, epochs=1)  # Adjust the number of epochs as needed
    # print("\nTraining with Adam optimizer completed.\n")
    #
    # # Display loss and accuracy for each epoch
    # print("Epoch results:")
    # for epoch in range(len(errors)):
    #     print(f"Epoch {epoch + 1}/{len(errors)} - Loss: {errors[epoch]:.4f} - Accuracy: {accuracies[epoch]:.4f}")


if __name__ == "__main__":
    main()
