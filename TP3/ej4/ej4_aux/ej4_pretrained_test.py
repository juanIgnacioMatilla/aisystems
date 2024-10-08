from matplotlib import pyplot as plt

from TP3.ej4.mnist_utils import print_training_errors, print_training_accuracies, \
    load_model, retrieve_preprocessed_mnist, graph_confusion_matrix, \
    confusion_matrix, graph_accuracy_v_epochs, accuracy_test_set, show_10prediction_errors, \
    load_preprocessed_mnist_with_noise, show_noisy_comparison, load_preprocessed_mnist


def main():
    # Load the preprocessed MNIST dataset
    (_, _), (x_test, y_test) = retrieve_preprocessed_mnist()

    # Load the trained model from a file
    # model_filename = 'trained_models/ADAM_10E_784_256_128_64_10.pkl'
    # loaded_mlp = load_model(model_filename)
    #
    # print_training_errors(loaded_mlp)
    # print_training_accuracies(loaded_mlp)
    # print(loaded_mlp.training_time)

    # confusion_matrix_256_128_64 = confusion_matrix(loaded_mlp, x_test, y_test, 10)
    # graph_confusion_matrix(confusion_matrix_256_128_64)

    # graph_accuracy_v_epochs(loaded_mlp)

    # # model_filename = 'trained_models/OVERFITTED_VANILLA_1000E_784_10_10.pkl'
    # model_filename = 'trained_models/UNDERFITTED_VANILLA_10E_784_10_10.pkl'
    # # model_filename = 'trained_models/UNDERFITTED_ADAM_1E_784_10_10.pkl'
    # loaded_mlp = load_model(model_filename)
    #
    # print_training_errors(loaded_mlp)
    # print_training_accuracies(loaded_mlp)
    # print(loaded_mlp.training_time)

    # Load the trained model from a file
    model_filename1 = 'trained_models/ADAM_10E_784_256_128_64_10.pkl'
    loaded_mlp1 = load_model(model_filename1)

    model_filename2 = 'trained_models/ADAM_100E_784_16_10_10.pkl'
    loaded_mlp2 = load_model(model_filename2)

    model_filename3 = 'trained_models/ADAM_3E_784_10_10.pkl'
    loaded_mlp3 = load_model(model_filename3)

    loaded_mlps = [loaded_mlp1, loaded_mlp2]

    # print("Time to train the model: ", loaded_mlp3.training_time)
    # print_training_errors(loaded_mlp)
    # print_training_accuracies(loaded_mlp)
    # # print(accuracy_test_set(loaded_mlp, x_test, y_test))
    #
    # # print()
    # #
    # show_10prediction_errors(loaded_mlp1, x_test, y_test)
    #
    # Load the preprocessed MNIST dataset with noise
    (x_train, y_train), (x_test_noisy1, y_test) = load_preprocessed_mnist_with_noise(0, 0.1)
    (x_train, y_train), (x_test_noisy2, y_test) = load_preprocessed_mnist_with_noise(0, 0.2)
    (x_train, y_train), (x_test_noisy3, y_test) = load_preprocessed_mnist_with_noise(0, 0.3)
    (x_train, y_train), (x_test_noisy4, y_test) = load_preprocessed_mnist_with_noise(0, 0.4)
    (x_train, y_train), (x_test_noisy5, y_test) = load_preprocessed_mnist_with_noise(0, 0.5)
    (x_train, y_train), (x_test_noisy6, y_test) = load_preprocessed_mnist_with_noise(0, 0.6)
    (x_train, y_train), (x_test_noisy7, y_test) = load_preprocessed_mnist_with_noise(0, 0.7)
    (x_train, y_train), (x_test_noisy8, y_test) = load_preprocessed_mnist_with_noise(0, 0.8)
    (x_train, y_train), (x_test_noisy9, y_test) = load_preprocessed_mnist_with_noise(0, 0.9)
    (x_train, y_train), (x_test, y_test) = load_preprocessed_mnist()

    # show_noisy_comparison(x_test, x_test_noisy1, 1)
    # show_noisy_comparison(x_test, x_test_noisy2, 1)
    # show_noisy_comparison(x_test, x_test_noisy3, 1)
    # show_noisy_comparison(x_test, x_test_noisy4, 1)
    # show_noisy_comparison(x_test, x_test_noisy6, 1)
    # show_noisy_comparison(x_test, x_test_noisy8, 1)

    confusion_matrix_256_128_64_1 = confusion_matrix(loaded_mlp1, x_test, y_test, 10)
    confusion_matrix_256_128_64_2 = confusion_matrix(loaded_mlp1, x_test_noisy1, y_test, 10)
    confusion_matrix_256_128_64_3 = confusion_matrix(loaded_mlp1, x_test_noisy2, y_test, 10)
    confusion_matrix_256_128_64_4 = confusion_matrix(loaded_mlp1, x_test_noisy3, y_test, 10)
    confusion_matrix_256_128_64_5 = confusion_matrix(loaded_mlp1, x_test_noisy4, y_test, 10)
    confusion_matrix_256_128_64_6 = confusion_matrix(loaded_mlp1, x_test_noisy5, y_test, 10)
    confusion_matrix_256_128_64_7 = confusion_matrix(loaded_mlp1, x_test_noisy6, y_test, 10)
    confusion_matrix_256_128_64_8 = confusion_matrix(loaded_mlp1, x_test_noisy7, y_test, 10)
    confusion_matrix_256_128_64_9 = confusion_matrix(loaded_mlp1, x_test_noisy8, y_test, 10)
    confusion_matrix_256_128_64_10 = confusion_matrix(loaded_mlp1, x_test_noisy9, y_test, 10)

    graph_confusion_matrix(confusion_matrix_256_128_64_1)
    graph_confusion_matrix(confusion_matrix_256_128_64_2)
    graph_confusion_matrix(confusion_matrix_256_128_64_3)
    graph_confusion_matrix(confusion_matrix_256_128_64_4)
    graph_confusion_matrix(confusion_matrix_256_128_64_5)
    graph_confusion_matrix(confusion_matrix_256_128_64_6)
    graph_confusion_matrix(confusion_matrix_256_128_64_7)
    graph_confusion_matrix(confusion_matrix_256_128_64_8)
    graph_confusion_matrix(confusion_matrix_256_128_64_9)
    graph_confusion_matrix(confusion_matrix_256_128_64_10)

    #
    # # Test the model on the different test sets
    # plt.figure(figsize=(10, 6))
    # plt.title("Accuracy vs Noise")
    # plt.xlabel("Noise level")
    # plt.ylabel("Accuracy")
    # accuracies1 = [accuracy_test_set(loaded_mlp1, x_test, y_test),
    #                accuracy_test_set(loaded_mlp1, x_test_noisy1, y_test),
    #                accuracy_test_set(loaded_mlp1, x_test_noisy2, y_test),
    #                accuracy_test_set(loaded_mlp1, x_test_noisy3, y_test),
    #                accuracy_test_set(loaded_mlp1, x_test_noisy4, y_test),
    #                accuracy_test_set(loaded_mlp1, x_test_noisy5, y_test),
    #                accuracy_test_set(loaded_mlp1, x_test_noisy6, y_test),
    #                accuracy_test_set(loaded_mlp1, x_test_noisy7, y_test),
    #                accuracy_test_set(loaded_mlp1, x_test_noisy8, y_test),
    #                accuracy_test_set(loaded_mlp1, x_test_noisy9, y_test)]
    # plt.bar(["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"], accuracies1, label="256-128-64", color='r')
    # plt.tight_layout()
    # plt.show()
    #
    # plt.figure(figsize=(10, 6))
    # plt.title("Accuracy vs Noise")
    # plt.xlabel("Noise level")
    # plt.ylabel("Accuracy")
    # accuracies2 = [accuracy_test_set(loaded_mlp2, x_test, y_test),
    #                  accuracy_test_set(loaded_mlp2, x_test_noisy1, y_test),
    #                  accuracy_test_set(loaded_mlp2, x_test_noisy2, y_test),
    #                  accuracy_test_set(loaded_mlp2, x_test_noisy3, y_test),
    #                  accuracy_test_set(loaded_mlp2, x_test_noisy4, y_test),
    #                  accuracy_test_set(loaded_mlp2, x_test_noisy5, y_test),
    #                  accuracy_test_set(loaded_mlp2, x_test_noisy6, y_test),
    #                  accuracy_test_set(loaded_mlp2, x_test_noisy7, y_test),
    #                  accuracy_test_set(loaded_mlp2, x_test_noisy8, y_test),
    #                  accuracy_test_set(loaded_mlp2, x_test_noisy9, y_test)]
    #
    #
    # plt.bar(["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"], accuracies2, label="16-10", color='b')
    # plt.tight_layout()
    # plt.show()
    #
    # plt.figure(figsize=(10, 6))
    # plt.title("Accuracy vs Noise")
    # plt.xlabel("Noise level")
    # plt.ylabel("Accuracy")
    # plt.ylim(0, 1)
    #
    # accuracies3 = [accuracy_test_set(loaded_mlp3, x_test, y_test),
    #                     accuracy_test_set(loaded_mlp3, x_test_noisy1, y_test),
    #                     accuracy_test_set(loaded_mlp3, x_test_noisy2, y_test),
    #                     accuracy_test_set(loaded_mlp3, x_test_noisy3, y_test),
    #                     accuracy_test_set(loaded_mlp3, x_test_noisy4, y_test),
    #                     accuracy_test_set(loaded_mlp3, x_test_noisy5, y_test),
    #                     accuracy_test_set(loaded_mlp3, x_test_noisy6, y_test),
    #                     accuracy_test_set(loaded_mlp3, x_test_noisy7, y_test),
    #                     accuracy_test_set(loaded_mlp3, x_test_noisy8, y_test),
    #                     accuracy_test_set(loaded_mlp3, x_test_noisy9, y_test)]
    # plt.bar(["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"], accuracies3, label="10", color='r')
    # plt.tight_layout()
    # plt.show()


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
