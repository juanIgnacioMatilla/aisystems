from TP3.ej4.mnist_utils import print_training_errors, print_training_accuracies, load_model


def main():

    # Load the trained model from a file
    model_filename = 'trained_models/ADAM_6E_784_64_10.pkl'
    loaded_mlp = load_model(model_filename)

    print("Time to train the model: ", loaded_mlp.training_time)
    print_training_errors(loaded_mlp)
    print_training_accuracies(loaded_mlp)


    # Load the trained model from a file
    model_filename = 'trained_models/ADAM_6E_784_10_10.pkl'
    loaded_mlp = load_model(model_filename)

    print("Time to train the model: ", loaded_mlp.training_time)
    print_training_errors(loaded_mlp)
    print_training_accuracies(loaded_mlp)

if __name__ == "__main__":
    main()
