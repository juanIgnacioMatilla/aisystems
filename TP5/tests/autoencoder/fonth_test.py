import random
import numpy as np
import matplotlib.pyplot as plt
import joblib  # Optional: For saving/loading models

from TP5.src.model.autoencoder import Autoencoder
from TP5.tests.utils import (
    to_bin_array, Font3, plot_max_error_char, interpolate_latent_vectors,
    plot_latent_space_with_interpolation, decode_and_plot_interpolated_chars
)


def run_training(X, layer_sizes, learning_rate, num_epochs, error_bar_interval=500):
    """
    Instantiate and train the autoencoder.

    Args:
        X (np.ndarray): Input data of shape (num_samples, input_size).
        layer_sizes (list): List specifying the size of each layer.
        learning_rate (float): Learning rate for training.
        num_epochs (int): Number of training epochs.
        error_bar_interval (int): Interval (in epochs) to record max pixel error.

    Returns:
        Autoencoder: The trained autoencoder instance.
        list: List of average loss per epoch.
        list: List of max pixel errors at specified intervals.
    """
    autoencoder = Autoencoder(layer_sizes)
    loss_history, max_pixel_error_history = autoencoder.train(
        X, learning_rate=learning_rate, num_epochs=num_epochs, error_bar_interval=error_bar_interval
    )
    return autoencoder, loss_history, max_pixel_error_history


if __name__ == "__main__":
    # Parameters for multiple runs
    NUM_RUNS = 10
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 4500
    ERROR_BAR_EPOCH_INTERVAL = 500  # Interval at which max pixel error is recorded

    # Prepare the input data
    X = []
    for char in Font3:
        bin_array = to_bin_array(char)
        bin_vector = bin_array.flatten()
        X.append(bin_vector)
    X = np.array(X).astype(float)

    # Define the architecture
    input_size = 35  # 7 rows * 5 columns
    hidden_layers = [60, 50, 30]  # Hidden layers before the latent layer
    latent_size = 2  # Latent space dimension
    layer_sizes = [input_size] + hidden_layers + [latent_size] + hidden_layers[::-1] + [input_size]

    # Initialize lists to store metrics and models for all runs
    all_loss_histories = []
    all_max_pixel_error_histories = []
    all_autoencoders = []

    for run in range(NUM_RUNS):
        # Optional: Set a unique seed for each run for reproducibility
        seed = 42 + run  # Example: 42, 43, 44, ...
        np.random.seed(seed)
        random.seed(seed)
        print(f"Starting run {run + 1}/{NUM_RUNS} with seed {seed}...")
        # Run training
        autoencoder, loss_history, max_pixel_error_history = run_training(
            X, layer_sizes, LEARNING_RATE, NUM_EPOCHS, ERROR_BAR_EPOCH_INTERVAL
        )
        all_autoencoders.append(autoencoder)
        all_loss_histories.append(loss_history)
        all_max_pixel_error_histories.append(max_pixel_error_history)
        print(f"Run {run + 1} completed.")

    # Convert lists to numpy arrays for easier computation
    all_loss_histories = np.array(all_loss_histories)  # Shape: (NUM_RUNS, NUM_EPOCHS)
    all_max_pixel_error_histories = np.array(all_max_pixel_error_histories)  # Shape: (NUM_RUNS, num_intervals)

    # Compute mean and standard deviation for loss
    mean_loss = np.mean(all_loss_histories, axis=0)
    std_loss = np.std(all_loss_histories, axis=0)

    # Compute mean and standard deviation for max pixel error
    mean_max_pixel_error = np.mean(all_max_pixel_error_histories, axis=0)
    std_max_pixel_error = np.std(all_max_pixel_error_histories, axis=0)

    # Plot the loss history with error bars (shaded area)
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    epochs = np.arange(1, NUM_EPOCHS + 1)
    plt.plot(epochs, mean_loss, label='Mean Loss', color='blue')
    plt.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, color='blue', alpha=0.2, label='Std Dev')
    plt.xlabel('Epochs')
    plt.ylabel('Average Loss')
    plt.title('Training Loss with Error Bars')
    plt.legend()

    # Plot the maximum pixel error history with error bars
    plt.subplot(1, 2, 2)
    interval_epochs = np.arange(ERROR_BAR_EPOCH_INTERVAL, NUM_EPOCHS + 1, ERROR_BAR_EPOCH_INTERVAL)
    plt.plot(interval_epochs, mean_max_pixel_error, marker='o', label='Mean Max Pixel Error', color='red')
    plt.fill_between(interval_epochs,
                     mean_max_pixel_error - std_max_pixel_error,
                     mean_max_pixel_error + std_max_pixel_error,
                     color='red', alpha=0.2, label='Std Dev')
    plt.xlabel('Epochs')
    plt.ylabel('Maximum Pixel Error')
    plt.title('Maximum Pixel Error Across Epochs with Error Bars')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Identify the run with the least final maximum pixel error
    final_max_pixel_errors = [history[-1] for history in all_max_pixel_error_histories]
    best_run_index = np.argmin(final_max_pixel_errors)
    best_autoencoder = all_autoencoders[best_run_index]
    print(f"Best run is Run {best_run_index + 1} with final maximum pixel error: {final_max_pixel_errors[best_run_index]}")

    # Optional: Save the best model
    # joblib.dump(best_autoencoder, 'best_autoencoder.pkl')
    # print("Best autoencoder model saved as 'best_autoencoder.pkl'.")

    # Reconstruct the characters using the best trained autoencoder
    reconstructed_chars = []
    for i in range(X.shape[0]):
        x = X[i].reshape(-1, 1)
        output = best_autoencoder.reconstruct(x)
        # Threshold the output
        reconstructed = (output > 0.5).astype(int)
        reconstructed_chars.append(reconstructed.flatten())

    # Compute pixel errors for all characters
    pixel_errors = []
    for i in range(X.shape[0]):
        x = X[i]
        reconstructed = reconstructed_chars[i]
        pixel_error = np.sum(np.abs(reconstructed - x))
        pixel_errors.append(pixel_error)

    # Identify the character with the maximum pixel error
    max_error_index = np.argmax(pixel_errors)
    print(f"Character with maximum pixel error is at index: {max_error_index}")

    # Plot the character with maximum pixel error
    plot_max_error_char(max_error_index, reconstructed_chars, X, pixel_errors)

    # Extract latent representations from the best run
    latent_representations = []
    for i in range(X.shape[0]):
        x = X[i].reshape(-1, 1)
        activations, _ = best_autoencoder.forward(x)
        latent = activations[4]  # Index of latent layer in activations
        latent_representations.append(latent.flatten())
    latent_representations = np.array(latent_representations)

    # Create labels for the characters
    labels = [chr(i) for i in range(96, 96 + Font3.shape[0])]

    # Plot the latent space
    plt.figure(figsize=(12, 10))
    plt.scatter(latent_representations[:, 0], latent_representations[:, 1], color='blue')

    # Annotate each point with its corresponding character
    for i, label in enumerate(labels):
        plt.annotate(label, (latent_representations[i, 0], latent_representations[i, 1]),
                     fontsize=12, textcoords="offset points", xytext=(0, 5), ha='center')

    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('Latent Space Representation of Characters')
    plt.grid(True)
    plt.show()

    # Select 5 random indices for visual inspection
    random_indices = random.sample(range(X.shape[0]), 5)

    fig, axes = plt.subplots(5, 2, figsize=(6, 12))

    for idx, (ax_orig, ax_recon) in zip(random_indices, axes):
        # Original character
        original = X[idx].reshape(7, 5)
        # Reconstructed character
        x = X[idx].reshape(-1, 1)
        output = best_autoencoder.reconstruct(x)
        reconstructed = (output > 0.5).astype(int).reshape(7, 5)

        # Plot original
        ax_orig.imshow(original, cmap='Greys')
        ax_orig.axis('off')
        ax_orig.set_title("Original")

        # Plot reconstructed
        ax_recon.imshow(reconstructed, cmap='Greys')
        ax_recon.axis('off')
        ax_recon.set_title("Reconstructed")

    plt.tight_layout()
    plt.show()

    # Generate a random latent vector and decode
    latent_vector = np.random.uniform(-1, 1, (2, 1))
    decoded_output = best_autoencoder.decode(latent_vector)
    reconstructed = (decoded_output > 0.5).astype(int).flatten()
    reconstructed_char = reconstructed.reshape(7, 5)

    # Interpolation between two latent vectors
    if X.shape[0] >= 3:  # Ensure there are at least 3 characters
        latent1 = latent_representations[1]
        latent2 = latent_representations[2]

        # Perform interpolation
        interpolated_latents = interpolate_latent_vectors(latent1, latent2, steps=10)
        print(f"Generated {len(interpolated_latents)} interpolated latent vectors.")

        # Plot the latent space with interpolation
        plot_latent_space_with_interpolation(
            latent_representations, labels, 1, 2, interpolated_latents, latent1, latent2
        )

        # Plot the interpolated characters
        decode_and_plot_interpolated_chars(best_autoencoder, interpolated_latents, title_prefix="Interp")
    else:
        print("Not enough characters to perform interpolation.")
