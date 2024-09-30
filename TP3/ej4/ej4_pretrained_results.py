from TP3.ej4.mnist_utils import print_training_errors, print_training_accuracies, load_model


def main():
    # Load the trained model from a file
    model_filename = 'trained_models/trained_mlp_ADAM_3E.pkl'
    loaded_mlp = load_model(model_filename)

    print_training_errors(loaded_mlp)
    print_training_accuracies(loaded_mlp)


if __name__ == "__main__":
    main()
