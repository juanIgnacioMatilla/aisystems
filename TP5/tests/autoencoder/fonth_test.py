import random

import numpy as np
import matplotlib.pyplot as plt

from TP5.src.model.autoencoder import Autoencoder
from TP5.tests.utils import to_bin_array, Font3, plot_max_error_char, interpolate_latent_vectors, \
    plot_latent_space_with_interpolation, decode_and_plot_interpolated_chars

if __name__ == "__main__":
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

    # Instantiate the autoencoder
    autoencoder = Autoencoder(layer_sizes)

    # Train the autoencoder
    loss_history, max_pixel_error_history = autoencoder.train(X, learning_rate=0.001, num_epochs=4500)

    # Plot the loss history
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.xlabel('Epochs')
    plt.ylabel('Average Loss')
    plt.title('Training Loss')

    # Adjust the x-axis for max pixel error to match the 500-epoch intervals
    epochs = list(range(500, 500 * len(max_pixel_error_history) + 1, 500))

    # Plot the maximum pixel error history
    plt.subplot(1, 2, 2)
    plt.plot(epochs, max_pixel_error_history, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Maximum Pixel Error')
    plt.title('Maximum Pixel Error Across Epochs')

    plt.tight_layout()
    plt.show()

    # Reconstruct the characters
    reconstructed_chars = []
    for i in range(X.shape[0]):
        x = X[i].reshape(-1, 1)
        output = autoencoder.reconstruct(x)
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

    # Plot the character
    plot_max_error_char(max_error_index, reconstructed_chars, X, pixel_errors)

    latent_vector = np.random.uniform(-1, 1, (2, 1))

    # Extract latent representations
    latent_representations = []
    for i in range(X.shape[0]):
        x = X[i].reshape(-1, 1)
        activations, _ = autoencoder.forward(x)
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

    # Select 5 random indices
    random_indices = random.sample(range(X.shape[0]), 5)

    fig, axes = plt.subplots(5, 2, figsize=(6, 12))

    for idx, (ax_orig, ax_recon) in zip(random_indices, axes):
        # Original character
        original = X[idx].reshape(7, 5)
        # Reconstructed character
        x = X[idx].reshape(-1, 1)
        output = autoencoder.reconstruct(x)
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

    # Ensure that the autoencoder has been trained
    # Generate a random latent vector
    latent_vector = np.random.uniform(-1, 1, (2, 1))

    # Decode the latent vector to get the reconstructed output
    decoded_output = autoencoder.decode(latent_vector)

    # Threshold the output to get binary values
    reconstructed = (decoded_output > 0.5).astype(int).flatten()

    # Reshape to 7x5 for plotting
    reconstructed_char = reconstructed.reshape(7, 5)
    # Generate and plot 5 new characters
    # generate_and_plot_multiple_chars(autoencoder, num_chars=5)

    # Extract the two selected latent vectors
    latent1 = latent_representations[1]
    latent2 = latent_representations[2]

    # Perform interpolation
    interpolated_latents = interpolate_latent_vectors(latent1, latent2, steps=10)
    print(f"Generated {len(interpolated_latents)} interpolated latent vectors.")

    # Plot the latent space with interpolation
    plot_latent_space_with_interpolation(latent_representations, labels, 1, 2, interpolated_latents, latent1, latent2)

    # Plot the interpolated characters
    decode_and_plot_interpolated_chars(autoencoder, interpolated_latents, title_prefix="Interp")
