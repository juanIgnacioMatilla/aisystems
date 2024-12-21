import time

import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset using Keras
from tensorflow.keras.datasets import mnist

from deep_learning.src.model.variational_autoencoder import VariationalAutoencoder
from deep_learning.tests.utils import plot_reconstructed_images, generate_images

if __name__ == "__main__":
    # Load the data
    (X_train_full, _), (X_test_full, _) = mnist.load_data()

    # Normalize and reshape the data
    X_train_full = X_train_full.astype(np.float32) / 255.0
    X_test_full = X_test_full.astype(np.float32) / 255.0
    X_train_full = X_train_full.reshape(-1, 28 * 28)
    X_test_full = X_test_full.reshape(-1, 28 * 28)

    # Reduce the dataset size for faster testing
    num_train_samples = 30000  # Adjust as needed
    X_train = X_train_full[:num_train_samples]

    num_test_samples = 6000  # Adjust as needed
    X_test = X_test_full[:num_test_samples]

    # Instantiate and train the VAE
    input_size = 784  # 28x28 pixels
    hidden_sizes = [512, 256]  # Hidden layers sizes
    latent_size = 2  # Latent space dimension

    vae = VariationalAutoencoder(input_size, hidden_sizes, latent_size)
    # Measure the time before training starts
    start_time = time.time()
    vae.train(X_train, batch_size=64, epochs=5, learning_rate=0.001)
    # Measure the time after training ends
    end_time = time.time()
    # Calculate and display the elapsed time
    elapsed_time = end_time - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds.")

    # Plot the loss function over epochs
    plt.figure()
    plt.plot(vae.loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.show()

    # Use a small subset of test data for visualization
    plot_reconstructed_images(vae, X_test[:10])

    generate_images(vae)
