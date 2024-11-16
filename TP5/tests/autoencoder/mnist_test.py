import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from tensorflow.keras.datasets import mnist
from TP5.src.model.autoencoder import Autoencoder
if __name__ == "__main__":
    # Load MNIST dataset with labels
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalize data to [-1, 1]
    X_train = (X_train.astype('float32') / 255.0) * 2 - 1
    X_test = (X_test.astype('float32') / 255.0) * 2 - 1

    # Flatten images
    X_train = X_train.reshape(-1, 28 * 28)
    X_test = X_test.reshape(-1, 28 * 28)

    # Define autoencoder architecture (symmetric)
    layer_sizes = [784, 128, 64, 32, 64, 128, 784]

    # Initialize autoencoder
    autoencoder = Autoencoder(layer_sizes)

    # Train autoencoder
    epochs = 20
    errors = autoencoder.train(X_train, epochs=epochs, batch_size=256, learning_rate=0.001)

    # Plot training error
    plt.plot(errors)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.title('Training Loss')
    plt.show()

    # Visualize latent space using t-SNE
    num_samples = 10000  # Number of samples to visualize
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    sample_images = X_test[indices]
    sample_labels = y_test[indices]

    # Get latent codes
    latent_codes = autoencoder.get_latent_codes(sample_images)

    # Reduce dimensionality to 2D using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    latent_codes_2d = tsne.fit_transform(latent_codes)

    # Plot the 2D latent space
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(latent_codes_2d[:, 0], latent_codes_2d[:, 1], c=sample_labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, ticks=range(10))
    plt.title('t-SNE Visualization of Latent Space')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()

    # Test reconstruction on test set
    num_images = 10
    np.random.seed(42)
    indices = np.random.choice(len(X_test), num_images, replace=False)
    sample_images = X_test[indices]
    reconstructed_images = autoencoder.reconstruct(sample_images)

    # Rescale images back to [0, 1] for display
    sample_images_display = (sample_images + 1) / 2
    reconstructed_images_display = (reconstructed_images + 1) / 2

    # Plot original and reconstructed images
    plt.figure(figsize=(20, 4))
    for i in range(num_images):
        # Original images
        ax = plt.subplot(2, num_images, i + 1)
        plt.imshow(sample_images_display[i].reshape(28, 28), cmap='gray')
        plt.title('Original')
        plt.axis('off')

        # Reconstructed images
        ax = plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(reconstructed_images_display[i].reshape(28, 28), cmap='gray')
        plt.title('Reconstructed')
        plt.axis('off')
    plt.show()