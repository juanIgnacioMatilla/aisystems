import dill
import tensorflow as tf
import numpy as np
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

    x_train_subset, y_train_subset = sample_data(x_train, y_train, size=sample_size)

    # Flatten the subset
    x_train_subset = x_train_subset.reshape(sample_size, 784)

    # One-hot encode the labels for the subsets (convert labels to 10-dimensional vectors)
    y_train_subset = tf.keras.utils.to_categorical(y_train_subset, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    return (x_train_subset, y_train_subset), (x_test, y_test)

def accuracy_test_set(loaded_mlp, x_test, y_test):
    # Test the model on the test set
    test_accuracies = []
    for i in range(len(x_test)):
        test_accuracies.append(np.argmax(loaded_mlp.predict(x_test[i])) == np.argmax(y_test[i]))

    return np.mean(test_accuracies)

def error_test_set(loaded_mlp, x_test, y_test):
    # Test the model on the test set
    test_errors = []
    for i in range(len(x_test)):
        # SSE error
        test_errors.append(0.5 * np.sum((loaded_mlp.predict(x_test[i]) - y_test[i]) ** 2))
    return np.sum(test_errors)

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

def graph_accuracy_v_epochs(loaded_mlp):
    # Create a new figure with adjusted size
    fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the figure size

    # Plot the accuracy values
    ax.plot(loaded_mlp.accuracies_by_epoch)

    # Set x-ticks and shift epoch labels by 1
    ax.set_xticks(np.arange(len(loaded_mlp.accuracies_by_epoch)))
    ax.set_xticklabels(np.arange(1, len(loaded_mlp.accuracies_by_epoch) + 1))  # Start at 1 instead of 0

    # Set titles and labels
    ax.set_title("Accuracy vs. Epochs", pad=20)  # Add padding to the title
    ax.set_xlabel("Epochs", labelpad=10)
    ax.set_ylabel("Accuracy", labelpad=10)

    # Show the plot
    plt.show()

def load_preprocessed_mnist_with_noise( noise_mean=0.0, noise_std=0.1):
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

    # Add Gaussian noise to the test set if specified
    x_test = add_gaussian_noise(x_test, noise_mean, noise_std)

    return (x_train, y_train), (x_test, y_test)

def add_gaussian_noise(x, mean=0.0, std=0.1):
    # Generate Gaussian noise with the same shape as x
    noise = np.random.normal(loc=mean, scale=std, size=x.shape)
    # Add the noise to x and ensure values remain in the [0, 1] range
    x_noisy = x + noise
    x_noisy = np.clip(x_noisy, 0.0, 1.0)
    return x_noisy

def show_noisy_comparison(x_test_original, x_test_noisy, num_samples=5):
    plt.figure(figsize=(10, 4))

    for i in range(num_samples):
        # Show original image
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(x_test_original[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.title('Original')

        # Show noisy image
        plt.subplot(2, num_samples, num_samples + i + 1)
        plt.imshow(x_test_noisy[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.title('Noisy')

    plt.tight_layout()
    plt.show()

def show_10prediction_errors(loaded_mlp, x_test, y_test):
    for i in range(1000):
        prediction = loaded_mlp.predict(x_test[i])
        predicted_digit = np.argmax(prediction)
        actual_digit = np.argmax(y_test[i])

        if predicted_digit != actual_digit:
            plt.figure(figsize=(10, 4))
            plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
            plt.axis('off')
            plt.title(f"Predicted: {predicted_digit}, Actual: {actual_digit}")
            plt.show()

    plt.tight_layout()
    plt.show()