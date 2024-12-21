import numpy as np
from matplotlib import pyplot as plt

from deep_learning.src.model.denoising_autoencoder import DenoisingAutoencoder
from deep_learning.tests.utils import add_noise, Font3, to_bin_array, plot_denoising_results

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
        # Extraer representaciones latentes de la mejor ejecución
        latent_representations = []
        for i in range(X.shape[0]):
            x = X[i].reshape(-1, 1)
            activations, _ = autoencoder.forward(x)
            latent = activations[len(hidden_layers) + 1]  # Índice de la capa latente en activations
            latent_representations.append(latent.flatten())
        latent_representations = np.array(latent_representations)

        # Crear etiquetas para los caracteres
        labels = [chr(i) for i in range(96, 96 + Font3.shape[0])]

        # Graficar el espacio latente
        plt.figure(figsize=(12, 10))
        plt.scatter(latent_representations[:, 0], latent_representations[:, 1], color='blue')

        # Anotar cada punto con su carácter correspondiente
        for i, label in enumerate(labels):
            plt.annotate(label, (latent_representations[i, 0], latent_representations[i, 1]),
                         fontsize=12, textcoords="offset points", xytext=(0, 5), ha='center')

        plt.xlabel('Dimensión Latente 1')
        plt.ylabel('Dimensión Latente 2')
        plt.title('Representación en el Espacio Latente de los Caracteres')
        plt.grid(True)
        plt.show()