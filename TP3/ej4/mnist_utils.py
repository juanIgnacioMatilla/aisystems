import dill
import tensorflow as tf
import numpy as np


def load_preprocessed_mnist():
    # Load and preprocess the MNIST dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize the data to [0, 1] range
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Flatten the 28x28 images into 1D arrays of 784 pixels
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)

    # One-hot encode the labels (convert labels to 10-dimensional vectors)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    return (x_train, y_train), (x_test, y_test)


# Store using dill the preprocessing of the MNIST dataset
def store_preprocessed_mnist(x_train, y_train, x_test, y_test):
    # Save the preprocessed MNIST dataset to a file
    with open('preprocessed_mnist.pkl', 'wb') as mnist_file:
        dill.dump(((x_train, y_train), (x_test, y_test)), mnist_file)


def retrieve_preprocessed_mnist():
    # Load the preprocessed MNIST dataset
    with open('preprocessed_mnist.pkl', 'rb') as mnist_file:
        return dill.load(mnist_file)


def store_model(adam_mlp, model_filename):
    # Save the trained model to a file
    with open(model_filename, 'wb') as model_file:
        dill.dump(adam_mlp, model_file)


def load_model(model_filename):
    # Load the trained model from a file
    with open(model_filename, 'rb') as model_file:
        return dill.load(model_file)


def load_preprocessed_mnist_sample(sample_size):
    # Load and preprocess the MNIST dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize the data to [0, 1] range
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Define a function to sample a subset of the data
    def sample_data(x, y, size):
        indices = np.random.choice(len(x), size, replace=False)  # Randomly choose indices
        return x[indices], y[indices]

    # Sample 10,000 training examples and 1,000 test examples from MNIST
    x_train_subset, y_train_subset = sample_data(x_train, y_train, size=sample_size)
    x_test_subset, y_test_subset = sample_data(x_test, y_test, size=sample_size)

    # Flatten the subset
    x_train_subset = x_train_subset.reshape(sample_size, 784)
    x_test_subset = x_test_subset.reshape(sample_size, 784)

    # One-hot encode the labels for the subsets (convert labels to 10-dimensional vectors)
    y_train_subset = tf.keras.utils.to_categorical(y_train_subset, num_classes=10)
    y_test_subset = tf.keras.utils.to_categorical(y_test_subset, num_classes=10)

    return (x_train_subset, y_train_subset), (x_test_subset, y_test_subset)


def print_n_predictions(loaded_mlp, x_test, y_test, n):
    # Display predictions for the first 10 test samples
    print("\nTest Predictions:")
    for i in range(n):  # Display predictions for the first 10 test samples
        predictions = loaded_mlp.predict(x_test[i])
        predicted_digit = np.argmax(predictions)  # Get the index of the highest probability (predicted digit)
        actual_digit = np.argmax(y_test[i])  # The actual digit from the test labels
        print(f"Test Sample {i}: Predicted: {predicted_digit}, Actual: {actual_digit}")
        print(f"Prediction probabilities: {predictions}")
        print()


def print_training_errors(loaded_mlp):
    # Display errors during training for some epochs
    print("\nTraining Errors:")
    for i, error in enumerate(loaded_mlp.errors_by_epoch):
        if i % 1 == 0:
            print(f"Error for epoch {i}: {error}")


def print_training_accuracies(loaded_mlp):
    # Display accuracies during training for some epochs
    print("\nTraining Accuracies:")
    for i, accuracy in enumerate(loaded_mlp.accuracies_by_epoch):
        if i % 1 == 0:
            print(f"Accuracy for epoch {i}: {accuracy}")
