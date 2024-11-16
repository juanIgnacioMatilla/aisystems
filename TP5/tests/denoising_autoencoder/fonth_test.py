import numpy as np
from matplotlib import pyplot as plt

from TP5.src.model.denoising_autoencoder import DenoisingAutoencoder
from TP5.tests.utils import add_noise, Font3, to_bin_array, plot_denoising_results

if __name__ == "__main__":
    X = []
    for char in Font3:
        bin_array = to_bin_array(char)
        bin_vector = bin_array.flatten()
        X.append(bin_vector)
    X = np.array(X).astype(float)
    # Define the architecture
    input_size = 35
    hidden_layers = [60, 50, 30]
    latent_size = 2
    layer_sizes = [input_size] + hidden_layers + [latent_size] + hidden_layers[::-1] + [input_size]

    # Instantiate the autoencoder
    autoencoder = DenoisingAutoencoder(layer_sizes)

    # Training parameters
    learning_rate = 0.001
    num_epochs = 5000

    # Noise levels to test
    noise_levels = [0.1, 0.3, 0.5]  # 10%, 30%, 50%

    for noise_level in noise_levels:
        print(f"\nTraining with noise level: {noise_level * 100}%")
        X_noisy = add_noise(X, noise_level)
        loss_history = autoencoder.train(X_noisy, X, learning_rate=learning_rate, num_epochs=num_epochs)

        # Plot the loss history
        plt.plot(loss_history)
        plt.xlabel('Epochs')
        plt.ylabel('Average Loss')
        plt.title(f'Training Loss (Noise Level: {noise_level * 100}%)')
        plt.show()

        # Reconstruct the images
        reconstructed_chars = []
        for i in range(X.shape[0]):
            x_noisy = X_noisy[i].reshape(-1, 1)
            output = autoencoder.reconstruct(x_noisy)
            reconstructed = (output > 0.5).astype(int)
            reconstructed_chars.append(reconstructed.flatten())
        num_examples = 5
        plot_denoising_results(X[:num_examples], X_noisy[:num_examples], reconstructed_chars[:num_examples],
                               num_examples)

        # Evaluate performance
        total_incorrect = 0
        for i in range(X.shape[0]):
            total_incorrect += np.sum(np.abs(reconstructed_chars[i] - X[i]))
        avg_incorrect = total_incorrect / X.shape[0]
        print(f"Average number of incorrect pixels per image: {avg_incorrect}")
