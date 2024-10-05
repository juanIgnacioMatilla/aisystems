import dill
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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


def store_model(mlp, model_filename):
    # Save the trained model to a file
    with open(model_filename, 'wb') as model_file:
        dill.dump(mlp, model_file)


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

def confusion_matrix(loaded_mlp, x_test, y_test, size):
    # Display predictions for all samples
    matrix = np.zeros((size, size), dtype=int)
    for i in range(len(x_test)):
        predictions = loaded_mlp.predict(x_test[i])
        predicted_digit = np.argmax(predictions)  # Get the index of the highest probability (predicted digit)
        actual_digit = np.argmax(y_test[i])  # The actual digit from the test labels
        matrix[actual_digit][predicted_digit] += 1
    return matrix

def graph_confusion_matrix(matrix):
    # Create a new figure with adjusted size
    fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the figure size

    # Display the confusion matrix as an image
    cax = ax.matshow(matrix, cmap='YlGnBu')  # Use 'YlGnBu' colormap to match sns

    # Add a colorbar for reference
    fig.colorbar(cax)

    # Loop over data dimensions and create text annotations
    for i in range(10):
        for j in range(10):
            if j == i:
                ax.text(j, i, int(matrix[i, j]), ha='center', va='center', color='white')
            else:
                ax.text(j, i, int(matrix[i, j]), ha='center', va='center', color='black')

    # Set titles and labels
    ax.set_title("Confusion Matrix", pad=20)  # Add padding to the title
    ax.set_xlabel("Predicted", labelpad=10)
    ax.set_ylabel("Actual", labelpad=10)

    # Set x and y axis labels to indicate digits
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))

    # Move x-axis ticks to the bottom
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.tick_bottom()

    # Turn on grid
    ax.grid(False)  # Disable default grid lines from matshow
    ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)  # Set minor ticks for grid
    ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)

    plt.show()






def print_training_errors(loaded_mlp):
    # Display errors during training for some epochs
    for i, error in enumerate(loaded_mlp.errors_by_epoch):
        if i % 1 == 0:
            print(f"Error for epoch {i}: {error}")


def print_training_accuracies(loaded_mlp):
    # Display accuracies during training for some epochs
    for i, accuracy in enumerate(loaded_mlp.accuracies_by_epoch):
        if i % 1 == 0:
            print(f"Accuracy for epoch {i}: {accuracy}")
