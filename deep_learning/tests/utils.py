# Function to convert encoded characters to binary arrays
import numpy as np
from matplotlib import pyplot as plt

# Prepare the input data
Font3 = np.array([
    [0x04, 0x04, 0x02, 0x00, 0x00, 0x00, 0x00],  # `
    [0x00, 0x0E, 0x01, 0x0D, 0x13, 0x13, 0x0D],  # a
    [0x10, 0x10, 0x10, 0x1C, 0x12, 0x12, 0x1C],  # b
    [0x00, 0x00, 0x00, 0x0E, 0x10, 0x10, 0x0E],  # c
    [0x01, 0x01, 0x01, 0x07, 0x09, 0x09, 0x07],  # d
    [0x00, 0x00, 0x0E, 0x11, 0x1F, 0x10, 0x0F],  # e
    [0x06, 0x09, 0x08, 0x1C, 0x08, 0x08, 0x08],  # f
    [0x0E, 0x11, 0x13, 0x0D, 0x01, 0x01, 0x0E],  # g
    [0x10, 0x10, 0x10, 0x16, 0x19, 0x11, 0x11],  # h
    [0x00, 0x04, 0x00, 0x0C, 0x04, 0x04, 0x0E],  # i
    [0x02, 0x00, 0x06, 0x02, 0x02, 0x12, 0x0C],  # j
    [0x10, 0x10, 0x12, 0x14, 0x18, 0x14, 0x12],  # k
    [0x0C, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],  # l
    [0x00, 0x00, 0x0A, 0x15, 0x15, 0x11, 0x11],  # m
    [0x00, 0x00, 0x16, 0x19, 0x11, 0x11, 0x11],  # n
    [0x00, 0x00, 0x0E, 0x11, 0x11, 0x11, 0x0E],  # o
    [0x00, 0x1C, 0x12, 0x12, 0x1C, 0x10, 0x10],  # p
    [0x00, 0x07, 0x09, 0x09, 0x07, 0x01, 0x01],  # q
    [0x00, 0x00, 0x16, 0x19, 0x10, 0x10, 0x10],  # r
    [0x00, 0x00, 0x0F, 0x10, 0x0E, 0x01, 0x1E],  # s
    [0x08, 0x08, 0x1C, 0x08, 0x08, 0x09, 0x06],  # t
    [0x00, 0x00, 0x11, 0x11, 0x11, 0x13, 0x0D],  # u
    [0x00, 0x00, 0x11, 0x11, 0x11, 0x0A, 0x04],  # v
    [0x00, 0x00, 0x11, 0x11, 0x15, 0x15, 0x0A],  # w
    [0x00, 0x00, 0x11, 0x0A, 0x04, 0x0A, 0x11],  # x
    [0x00, 0x11, 0x11, 0x0F, 0x01, 0x11, 0x0E],  # y
    [0x00, 0x00, 0x1F, 0x02, 0x04, 0x08, 0x1F],  # z
    [0x06, 0x08, 0x08, 0x10, 0x08, 0x08, 0x06],  # {
    [0x04, 0x04, 0x04, 0x00, 0x04, 0x04, 0x04],  # |
    [0x0C, 0x02, 0x02, 0x01, 0x02, 0x02, 0x0C],  # }
    [0x08, 0x15, 0x02, 0x00, 0x00, 0x00, 0x00],  # ~
    [0x1F, 0x1F, 0x1F, 0x1F, 0x1F, 0x1F, 0x1F]  # DEL
])


# Plot the original and reconstructed character with the maximum pixel error
def plot_max_error_char(index, reconstructed_chars, X, pixel_errors):
    fig, axes = plt.subplots(1, 2, figsize=(6, 4))
    # Original character
    axes[0].imshow(X[index].reshape(7, 5), cmap='Greys')
    axes[0].axis('off')
    axes[0].set_title("Original")
    # Reconstructed character
    axes[1].imshow(reconstructed_chars[index].reshape(7, 5), cmap='Greys')
    axes[1].axis('off')
    axes[1].set_title("Reconstructed")
    plt.suptitle(f'Character with Maximum Pixel Error ({pixel_errors[index]} pixels)')
    plt.show()

def to_bin_array(encoded_character):
    bin_array = np.zeros((7, 5), dtype=int)
    for row in range(7):
        current_row = encoded_character[row]
        for col in range(5):
            bin_array[row][4 - col] = current_row & 1
            current_row >>= 1
    return bin_array

def plot_generated_char(char_matrix, title="Generated Character"):
    """
    Plot a 7x5 binary character matrix.
    Args:
        char_matrix (numpy.ndarray): A 7x5 binary matrix
        title (str): Title of the plot
    """
    plt.figure(figsize=(2, 3))
    plt.imshow(char_matrix, cmap='Greys', interpolation='nearest')
    plt.axis('off')
    plt.title(title)
    plt.show()


def generate_and_plot_multiple_chars(autoencoder, num_chars=5):
    for i in range(num_chars):
        latent_vector = np.random.uniform(-1, 1, (2, 1))
        decoded_output = autoencoder.decode(latent_vector)
        reconstructed = (decoded_output > 0.5).astype(int).flatten()
        reconstructed_char = reconstructed.reshape(7, 5)
        plot_generated_char(reconstructed_char, title=f"Generated Character {i+1}")


def interpolate_latent_vectors(latent1, latent2, steps=10):
    """
    Linearly interpolate between two latent vectors.

    Args:
        latent1 (numpy.ndarray): First latent vector of shape (2,)
        latent2 (numpy.ndarray): Second latent vector of shape (2,)
        steps (int): Number of interpolation steps

    Returns:
        list of numpy.ndarray: List of interpolated latent vectors
    """
    interpolated = []
    for alpha in np.linspace(0, 1, steps):
        latent_interp = (1 - alpha) * latent1 + alpha * latent2
        interpolated.append(latent_interp)
    return interpolated
def plot_latent_space_with_interpolation(
        latent_representations,
        labels,
        index1,
        index2,
        interpolated_latents,
        latent1, latent2
        ):
    """
    Plot the latent space, highlighting two original points and their interpolations.

    Args:
        latent_representations (numpy.ndarray): Array of latent vectors of shape (num_chars, 2)
        labels (list): List of character labels
        index1 (int): Index of the first selected character
        index2 (int): Index of the second selected character
        interpolated_latents (list of numpy.ndarray): List of interpolated latent vectors
    """
    plt.figure(figsize=(10, 8))

    # Plot all points
    plt.scatter(latent_representations[:, 0], latent_representations[:, 1], color='lightblue',
                label='Original Characters')

    # Highlight the two selected points
    plt.scatter(latent_representations[index1, 0], latent_representations[index1, 1], color='red',
                label=f"'{labels[index1]}'")
    plt.scatter(latent_representations[index2, 0], latent_representations[index2, 1], color='green',
                label=f"'{labels[index2]}'")

    # Plot interpolated points
    interpolated_latents = np.array(interpolated_latents)
    plt.scatter(interpolated_latents[:, 0], interpolated_latents[:, 1], color='orange', label='Interpolated Points')

    # Draw lines between original points and interpolations
    plt.plot([latent1[0], latent2[0]], [latent1[1], latent2[1]], 'k--', label='Interpolation Path')

    # Annotate the original points
    plt.annotate(labels[index1], (latent_representations[index1, 0], latent_representations[index1, 1]),
                 textcoords="offset points", xytext=(0, 10), ha='center', color='red', fontsize=12)
    plt.annotate(labels[index2], (latent_representations[index2, 0], latent_representations[index2, 1]),
                 textcoords="offset points", xytext=(0, 10), ha='center', color='green', fontsize=12)

    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('Latent Space with Interpolation Path')
    plt.legend()
    plt.grid(True)
    plt.show()

def decode_and_plot_interpolated_chars(autoencoder, interpolated_latents, title_prefix="Interpolated"):
    """
    Decode interpolated latent vectors and plot the corresponding characters.

    Args:
        autoencoder (Autoencoder): Trained autoencoder instance
        interpolated_latents (list of numpy.ndarray): List of interpolated latent vectors
        title_prefix (str): Prefix for the plot titles
    """
    num_steps = len(interpolated_latents)
    fig, axes = plt.subplots(1, num_steps, figsize=(num_steps * 2, 3))

    for i, latent in enumerate(interpolated_latents):
        latent_vector = latent.reshape(-1, 1)
        decoded_output = autoencoder.decode(latent_vector)
        reconstructed = (decoded_output > 0.5).astype(int).flatten()
        reconstructed_char = reconstructed.reshape(7, 5)

        ax = axes[i]
        ax.imshow(reconstructed_char, cmap='Greys', interpolation='nearest')
        ax.axis('off')
        ax.set_title(f"{title_prefix} {i + 1}")

    plt.suptitle('Interpolated Characters')
    plt.tight_layout()
    plt.show()

def add_noise(X, noise_level):
    """
    Add salt-and-pepper noise to the binary images.
    noise_level: float between 0 and 1 indicating the fraction of pixels to corrupt.
    """
    X_noisy = X.copy()
    num_samples, num_features = X.shape
    num_noisy = int(noise_level * num_features)
    for i in range(num_samples):
        # Randomly choose indices to flip
        noisy_indices = np.random.choice(num_features, num_noisy, replace=False)
        X_noisy[i, noisy_indices] = 1 - X_noisy[i, noisy_indices]  # Flip bits
    return X_noisy

# Visualize results
def plot_denoising_results(original_chars, noisy_chars, reconstructed_chars, num_examples):
    fig, axes = plt.subplots(3, num_examples, figsize=(num_examples * 2, 6))
    for i in range(num_examples):
        axes[0, i].imshow(original_chars[i].reshape(7, 5), cmap='Greys')
        axes[0, i].axis('off')
        axes[0, i].set_title("Original")
        axes[1, i].imshow(noisy_chars[i].reshape(7, 5), cmap='Greys')
        axes[1, i].axis('off')
        axes[1, i].set_title("Noisy")
        axes[2, i].imshow(reconstructed_chars[i].reshape(7, 5), cmap='Greys')
        axes[2, i].axis('off')
        axes[2, i].set_title("Reconstructed")
    plt.tight_layout()
    plt.show()


# Visualize reconstructed images
def plot_reconstructed_images(vae, X, num_images=10):
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))
    for i in range(num_images):
        x = X[i].reshape(-1, 1)
        y = vae.reconstruct(x)
        axes[0, i].imshow(x.reshape(28, 28), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title("Original")
        axes[1, i].imshow(y.reshape(28, 28), cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title("Reconstructed")
    plt.tight_layout()
    plt.show()

# Generate new images by sampling from the latent space
def generate_images(vae, num_images=10):
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 2))
    for i in range(num_images):
        z = np.random.randn(vae.latent_size, 1)
        # Decoder forward pass
        a = z
        for j in range(len(vae.hidden_sizes)):
            W = vae.weights[f'decoder_W{j+1}']
            b = vae.biases[f'decoder_b{j+1}']
            z_dec = np.dot(W, a) + b
            a = vae.relu(z_dec)
        W_out = vae.weights['decoder_W_out']
        b_out = vae.biases['decoder_b_out']
        y = np.dot(W_out, a) + b_out
        y = vae.sigmoid(y)
        axes[i].imshow(y.reshape(28, 28), cmap='gray')
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()


