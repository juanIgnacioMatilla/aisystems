import re

import dill
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
from numpy.fft import fft2, fftshift
from PIL import Image
import imagehash

mnist = tf.keras.datasets.mnist


def binarize_images(images, threshold=0.5):
    return (images > threshold).astype(np.float32)


def load_mnist_data_split():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Normalizar y binarizar las imágenes
    x_train = x_train.astype('float32') / 255
    x_train = binarize_images(x_train)
    x_train = x_train.reshape((x_train.shape[0], -1))

    x_test = x_test.astype('float32') / 255
    x_test = binarize_images(x_test)
    x_test = x_test.reshape((x_test.shape[0], -1))

    return x_train, y_train, x_test, y_test


def load_mnist_data_split_sample(sample_size):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Normalizar y binarizar las imágenes
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    x_train = binarize_images(x_train)
    x_test = binarize_images(x_test)

    # Define a function to sample a subset of the data
    def sample_data(x, y, size):
        indices = np.random.choice(len(x), size, replace=False)  # Randomly choose indices
        return x[indices], y[indices]

    x_train_subset, y_train_subset = sample_data(x_train, y_train, size=sample_size)

    x_train_subset = x_train_subset.reshape((x_train_subset.shape[0], -1))
    x_test = x_test.reshape((x_test.shape[0], -1))

    return x_train_subset, y_train_subset, x_test, y_test


def add_noise_to_image(image, noise_level=0.1):
    """
    Agrega ruido binomial a una imagen binarizada.

    :param image: Vector de la imagen original (valores 0 o 1).
    :param noise_level: Porcentaje de píxeles a los que se les agregará ruido.
    :return: Imagen con ruido.
    """
    noisy_image = image.copy()
    n_pixels = len(image)
    n_noisy = int(noise_level * n_pixels)
    noisy_indices = np.random.choice(n_pixels, n_noisy, replace=False)
    noisy_image[noisy_indices] = 1 - noisy_image[noisy_indices]
    return noisy_image


def mean_squared_error(original, reconstructed):
    """
    Calcula el Error Cuadrático Medio (MSE) entre las imágenes originales y reconstruidas.

    :param original: Array de forma (n_samples, n_features) de las imágenes originales.
    :param reconstructed: Array de forma (n_samples, n_features) de las imágenes reconstruidas.
    :return: MSE promedio.
    """
    return np.mean((original - reconstructed) ** 2)


def pixelwise_error(original, reconstructed):
    """
    Penalizes extra and missing pixels between the original and reconstructed images.
    """
    missing_pixels = np.sum((original == 1) & (reconstructed == 0))
    extra_pixels = np.sum((original == 0) & (reconstructed == 1))
    return (missing_pixels + extra_pixels) / original.size


def hamming_loss(original, reconstructed):
    """
    Hamming loss is a pixel-wise mismatch penalty between the original and reconstructed.
    """
    return np.sum(original != reconstructed) / original.size


# Cargar y preprocesar los datos de MNIST
def load_mnist_data_no_binarized():
    (x_train, _), (_, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255
    x_train = x_train.reshape(60000, 784)
    return x_train


def store_model(mlp, model_filename):
    # Save the trained model to a file
    with open(model_filename, 'wb') as model_file:
        dill.dump(mlp, model_file)


def load_model(model_filename):
    # Load the trained model from a file
    with open(model_filename, 'rb') as model_file:
        return dill.load(model_file)


def compare_images(original, reconstructed):
    # Ensure the images are normalized (range [0, 1]) or (0, 255)
    original = (original * 255).astype('uint8') if original.max() <= 1 else original
    reconstructed = (reconstructed * 255).astype('uint8') if reconstructed.max() <= 1 else reconstructed

    # Reshape if images are flattened
    if original.ndim == 1:  # Assuming images are square
        side_length = int(np.sqrt(original.size))
        original = original.reshape((side_length, side_length))

    if reconstructed.ndim == 1:  # Assuming images are square
        side_length = int(np.sqrt(reconstructed.size))
        reconstructed = reconstructed.reshape((side_length, side_length))

    comparison_results = {}

    # 1. Structural Similarity Index (SSIM)
    ssim_value, _ = ssim(original, reconstructed, full=True)
    comparison_results['SSIM'] = ssim_value

    # 2. Mean Squared Error (MSE)
    mse_value = np.mean((original - reconstructed) ** 2)
    comparison_results['MSE'] = mse_value

    # 3. Fourier Transform (Frequency Domain) Comparison
    fft_original = fftshift(fft2(original))
    fft_reconstructed = fftshift(fft2(reconstructed))
    magnitude_original = np.log(np.abs(fft_original) + 1)
    magnitude_reconstructed = np.log(np.abs(fft_reconstructed) + 1)
    mse_frequency = np.mean((magnitude_original - magnitude_reconstructed) ** 2)
    comparison_results['MSE in Frequency Domain'] = mse_frequency

    # 4. Perceptual Hash (pHash)
    original_pil = Image.fromarray(original)
    reconstructed_pil = Image.fromarray(reconstructed)
    hash_original = imagehash.phash(original_pil)
    hash_reconstructed = imagehash.phash(reconstructed_pil)
    hamming_distance = hash_original - hash_reconstructed
    comparison_results['Perceptual Hash Hamming Distance'] = hamming_distance

    return comparison_results


def plot_accuracy_vs_noise(x_test, model, noise_levels, num_runs=3):
    """
    Calculate and plot the accuracy (Mean SSIM) vs. noise level for a given model.

    Parameters:
    - x_test: Array of test images.
    - model: The trained model for reconstruction (e.g., DBN).
    - noise_levels: Array of noise levels to test.
    - num_runs: Number of runs to average SSIM values for each noise level.
    """
    mean_ssims = []  # To store mean SSIM for each noise level
    std_ssims = []  # To store standard deviation of SSIM for each noise level

    # Loop through each noise level and calculate SSIM statistics
    for noise_level in noise_levels:
        # Use evaluate_model_ssim to get all SSIM values for this noise level
        ssim_values = evaluate_model_ssim(model, x_test, noise_level, num_runs=num_runs)

        # Use calculate_ssim_statistics to get mean and std dev of SSIM values
        mean_ssim, std_ssim = calculate_ssim_statistics(ssim_values)

        # Store the results for plotting
        mean_ssims.append(mean_ssim)
        std_ssims.append(std_ssim)

    # Plot Accuracy vs. Noise as a bar graph with error bars
    plt.figure(figsize=(10, 6))
    plt.bar(noise_levels, mean_ssims, yerr=std_ssims, width=0.01, color='b', alpha=0.7, capsize=5)
    plt.title('Accuracy (Mean SSIM) vs. Noise Level for ' + model.__class__.__name__)
    plt.xlabel('Noise Level')
    plt.ylabel('Mean SSIM')
    plt.xticks(noise_levels)
    plt.ylim(0, 1)  # Assuming SSIM values are normalized between 0 and 1
    plt.grid(axis='y')
    plt.show()


def evaluate_model_ssim(model, x_test, noise_level, num_runs=3):
    """
    Evaluates SSIM values for a given model and noise level.

    Args:
        model: The trained model used for reconstruction.
        x_test: Test images.
        noise_level: The level of noise to add to test images.
        num_runs: The number of runs to average over.

    Returns:
        List of SSIM values for all test images across runs.
    """
    ssim_values = []

    for _ in range(num_runs):
        for i in range(len(x_test)):
            test_image = x_test[i]

            # Add noise to the test image
            noisy_image = add_noise_to_image(test_image, noise_level=noise_level)
            noisy_image_reshaped = noisy_image.reshape(1, -1)

            # Reconstruct with the model
            reconstructed = model.reconstruct(noisy_image_reshaped).flatten()

            # Reshape both images to 28x28 and ensure float type
            test_image_reshaped = test_image.reshape(28, 28).astype(np.float32)
            reconstructed_reshaped = reconstructed.reshape(28, 28).astype(np.float32)

            # Calculate SSIM and store it
            ssim_value = ssim(test_image_reshaped, reconstructed_reshaped, data_range=1.0)
            ssim_values.append(ssim_value)

    return ssim_values


def calculate_ssim_statistics(ssim_values):
    """
    Calculates mean and standard deviation of SSIM values.

    Args:
        ssim_values: List of SSIM values.

    Returns:
        Tuple of mean and standard deviation of SSIM values.
    """
    mean_ssim = np.mean(ssim_values)
    std_ssim = np.std(ssim_values)
    return mean_ssim, std_ssim


def plot_model_accuracies(model_tags, mean_ssims, std_ssims):
        plt.figure(figsize=(12, 6))

        # Generate a bar plot for each model with error bars for standard deviation
        x_pos = np.arange(len(model_tags))
        plt.bar(x_pos, mean_ssims, yerr=std_ssims, capsize=5, color='skyblue', edgecolor='grey')

        plt.xticks(x_pos, model_tags, rotation=45, ha="right")
        plt.title('Model Accuracy Comparison (Mean SSIM)')
        plt.xlabel('Models')
        plt.ylabel('Mean SSIM')
        plt.ylim(0, 1)  # Assuming SSIM is between 0 and 1
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()

def extract_tags(filenames):
    tags = []
    for filename in filenames:
        # Use regex to find DBN_Ex and RBM_Ex patterns
        match = re.search(r'(DBN|RBM)_(E\d+)', filename)
        if match:
            tags.append(match.group(0))  # Add the full match (DBN_Ex or RBM_Ex)
    return tags