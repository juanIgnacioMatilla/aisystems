import numpy as np
import tensorflow as tf
from TP3.src.model.multilayer_perceptron.adam.adam_multi_layer_perceptron import AdamMultiLayerPerceptron

def main():
    print(tf.__version__)
    # Load and preprocess the MNIST dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize the data to [0, 1] range
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Define a function to sample a subset of the data
    def sample_data(x, y, sample_size):
        indices = np.random.choice(len(x), sample_size, replace=False)  # Randomly choose indices
        return x[indices], y[indices]

    # Sample 10,000 training examples and 1,000 test examples from MNIST
    x_train_subset, y_train_subset = sample_data(x_train, y_train, sample_size=1000)
    x_test_subset, y_test_subset = sample_data(x_test, y_test, sample_size=1000)

    #Flatten the subset
    x_train_subset = x_train_subset.reshape(1000, 784)
    x_test_subset= x_test_subset.reshape(1000, 784)

    # One-hot encode the labels for the subsets (convert labels to 10-dimensional vectors)
    y_train_subset = tf.keras.utils.to_categorical(y_train_subset, num_classes=10)
    y_test_subset = tf.keras.utils.to_categorical(y_test_subset, num_classes=10)


    # Flatten the 28x28 images into 1D arrays of 784 pixels
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)

    # One-hot encode the labels (convert labels to 10-dimensional vectors)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    # Define the MLP structure
    # Input size is 784 (flattened 28x28 images), hidden layer with 64 neurons, output layer with 10 neurons
    adam_mlp = AdamMultiLayerPerceptron(layers_structure=[784, 64, 10])

    # Train the MLP
    # errors = adam_mlp.train(x_train, y_train, epochs=10)  # Adjust the number of epochs as needed
    errors = adam_mlp.train(x_train_subset, y_train_subset, epochs=2)  # Adjust the number of epochs as needed
    print("Training with Adam optimizer completed.")

    # Test the MLP on the test dataset and show predictions for some digits
    print("\nTest Predictions:")
    for i in range(10):  # Display predictions for the first 10 test samples
        predictions = adam_mlp.predict(x_test[i])
        predicted_digit = np.argmax(predictions)  # Get the index of the highest probability (predicted digit)
        actual_digit = np.argmax(y_test[i])  # The actual digit from the test labels
        print(f"Test Sample {i}: Predicted: {predicted_digit}, Actual: {actual_digit}")
        print(f"Prediction probabilities: {predictions}")
        print()

    # Display errors during training for some epochs
    print("\nTraining Errors:")
    for i, error in enumerate(errors):
        if i % 1 == 0:  # Change the modulus if you want to print errors for specific epochs
            print(f"Error for epoch {i}: {error}")

if __name__ == "__main__":
    main()
