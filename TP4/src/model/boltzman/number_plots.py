import numpy as np
from matplotlib import pyplot as plt

from TP4.src.model.boltzman.boltzmann_utils import add_noise_to_image, compare_images, mean_squared_error, \
    pixelwise_error, hamming_loss, binarize_images, load_model


def plot_noisy_comparison(x_test, rbm_individual, dbn):
    # Modify your loop to include the plotting logic
    for i in range(15):
        # Select a test image
        test_image = x_test[i]

        # Add noise to the test image
        noisy_image = add_noise_to_image(test_image, noise_level=0.07)
        noisy_image_reshaped = noisy_image.reshape(1, -1)

        # Reconstruct with RBM and DBN (with and without noise)
        reconstructed_rbm = rbm_individual.reconstruct(noisy_image_reshaped).flatten()
        reconstructed_dbn = dbn.reconstruct(noisy_image_reshaped).flatten()
        reconstructed_rbm_no_noise = rbm_individual.reconstruct(test_image.reshape(1, -1)).flatten()
        reconstructed_dbn_no_noise = dbn.reconstruct(test_image.reshape(1, -1)).flatten()

        # Prepare comparison results for each reconstruction
        metrics_rbm = compare_images(test_image, reconstructed_rbm)
        metrics_dbn = compare_images(test_image, reconstructed_dbn)
        metrics_rbm_no_noise = compare_images(test_image, reconstructed_rbm_no_noise)
        metrics_dbn_no_noise = compare_images(test_image, reconstructed_dbn_no_noise)

        # Prepare images for visualization
        images_top = [
            test_image.reshape(28, 28),  # Original
            reconstructed_rbm_no_noise.reshape(28, 28),  # RBM no noise
            reconstructed_dbn_no_noise.reshape(28, 28)  # DBN no noise
        ]

        images_bottom = [
            noisy_image.reshape(28, 28),  # Noisy
            reconstructed_rbm.reshape(28, 28),  # RBM
            reconstructed_dbn.reshape(28, 28)  # DBN
        ]

        titles_top = [
            "Original",
            f"RBM\nSSIM: {metrics_rbm_no_noise['SSIM']:.2f}, MSE: {metrics_rbm_no_noise['MSE']:.2f}",
            f"DBN\nSSIM: {metrics_dbn_no_noise['SSIM']:.2f}, MSE: {metrics_dbn_no_noise['MSE']:.2f}"
        ]

        titles_bottom = [
            f"Noisy (level=0.07)",
            f"RBM noise\nSSIM: {metrics_rbm['SSIM']:.2f}, MSE: {metrics_rbm['MSE']:.2f}",
            f"DBN noise\nSSIM: {metrics_dbn['SSIM']:.2f}, MSE: {metrics_dbn['MSE']:.2f}"
        ]

        # Create a 2-row plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Top row
        for idx, ax in enumerate(axes[0]):
            ax.imshow(images_top[idx], cmap='gray')
            ax.set_title(titles_top[idx], fontsize=12)
            ax.axis('off')

        # Bottom row
        for idx, ax in enumerate(axes[1]):
            ax.imshow(images_bottom[idx], cmap='gray')
            ax.set_title(titles_bottom[idx], fontsize=12)
            ax.axis('off')

        plt.tight_layout()
        plt.show()

def plot_all_reconstructions(x_test, rbm_individual, dbn):
    for i in range(15):
        # Seleccionar una imagen de prueba
        test_image = x_test[i]

        mse_rbm = mean_squared_error(test_image, rbm_individual.reconstruct(test_image))
        pixelwise_error_rbm = pixelwise_error(test_image, rbm_individual.reconstruct(test_image))
        hamming_loss_rbm = hamming_loss(test_image, rbm_individual.reconstruct(test_image))
        mse_rbm_binarized = mean_squared_error(test_image, binarize_images(rbm_individual.reconstruct(test_image)))
        pixelwise_error_rbm_binarized = pixelwise_error(test_image,
                                                        binarize_images(rbm_individual.reconstruct(test_image)))
        hamming_loss_rbm_binarized = hamming_loss(test_image, binarize_images(rbm_individual.reconstruct(test_image)))

        mse_dbn = mean_squared_error(test_image, dbn.reconstruct(test_image))
        pixelwise_error_dbn = pixelwise_error(test_image, dbn.reconstruct(test_image))
        hamming_loss_dbn = hamming_loss(test_image, dbn.reconstruct(test_image))
        mse_dbn_binarized = mean_squared_error(test_image, binarize_images(dbn.reconstruct(test_image)))
        pixelwise_error_dbn_binarized = pixelwise_error(test_image, binarize_images(dbn.reconstruct(test_image)))
        hamming_loss_dbn_binarized = hamming_loss(test_image, binarize_images(dbn.reconstruct(test_image)))

        # Agregar ruido a la imagen de prueba
        noisy_image = add_noise_to_image(x_test[i], noise_level=0.07)
        noisy_image_reshaped = noisy_image.reshape(1, -1)

        # Reconstruir con RBM individual
        reconstructed_rbm = rbm_individual.reconstruct(noisy_image_reshaped).flatten()
        reconstructed_rbm_no_noise = rbm_individual.reconstruct(test_image.reshape(1, -1)).flatten()

        # Reconstruir con DBN
        reconstructed_dbn = dbn.reconstruct(noisy_image_reshaped).flatten()
        reconstructed_dbn_no_noise = dbn.reconstruct(test_image.reshape(1, -1)).flatten()

        # Visualizar los resultados
        plt.figure(figsize=(15, 6))

        plt.subplot(1, 10, 1)
        plt.title("Original")
        plt.imshow(test_image.reshape(28, 28), cmap='gray')
        plt.axis('off')

        plt.subplot(1, 10, 2)
        plt.title("Con Ruido")
        plt.imshow(noisy_image.reshape(28, 28), cmap='gray')
        plt.axis('off')

        plt.subplot(1, 10, 3)
        plt.title("RBM con Ruido")
        plt.imshow(reconstructed_rbm.reshape(28, 28), cmap='gray')
        plt.axis('off')

        plt.subplot(1, 10, 4)
        plt.title("Binarized RBM con Ruido")
        plt.imshow(binarize_images(reconstructed_rbm).reshape(28, 28), cmap='gray')
        plt.axis('off')

        plt.subplot(1, 10, 5)
        plt.title("RBM sin Ruido")
        plt.imshow(reconstructed_rbm_no_noise.reshape(28, 28), cmap='gray')
        plt.axis('off')

        plt.subplot(1, 10, 6)
        plt.title("Binarized RBM sin Ruido")
        plt.imshow(binarize_images(reconstructed_rbm_no_noise).reshape(28, 28), cmap='gray')
        plt.axis('off')

        plt.subplot(1, 10, 7)
        plt.title("DBN Ruido")
        plt.imshow(reconstructed_dbn.reshape(28, 28), cmap='gray')
        plt.axis('off')

        plt.subplot(1, 10, 8)
        plt.title("Binarized DBN con Ruido")
        plt.imshow(binarize_images(reconstructed_dbn).reshape(28, 28), cmap='gray')
        plt.axis('off')

        plt.subplot(1, 10, 9)
        plt.title("DBN sin Ruido")
        plt.imshow(reconstructed_dbn_no_noise.reshape(28, 28), cmap='gray')
        plt.axis('off')

        plt.subplot(1, 10, 10)
        plt.title("Binarized DBN sin Ruido")
        plt.imshow(binarize_images(reconstructed_dbn_no_noise).reshape(28, 28), cmap='gray')
        plt.axis('off')

        mlp_test_model = load_model('../../../../TP3/ej4/trained_models/XAVIER_ADAM_10E_784_256_128_64_10.pkl')
        prediction_reconstructed_rbm = mlp_test_model.predict(reconstructed_rbm.reshape(1, -1))
        prediction_reconstructed_rbm_no_noise = mlp_test_model.predict(reconstructed_rbm_no_noise.reshape(1, -1))
        prediction_reconstructed_dbn = mlp_test_model.predict(reconstructed_dbn.reshape(1, -1))
        prediction_reconstructed_dbn_no_noise = mlp_test_model.predict(reconstructed_dbn_no_noise.reshape(1, -1))
        prediction_reconstructed_rbm_binarized = mlp_test_model.predict(
            binarize_images(reconstructed_rbm).reshape(1, -1))
        prediction_reconstructed_rbm_no_noise_binarized = mlp_test_model.predict(
            binarize_images(reconstructed_rbm_no_noise).reshape(1, -1))
        prediction_reconstructed_dbn_binarized = mlp_test_model.predict(
            binarize_images(reconstructed_dbn).reshape(1, -1))
        prediction_reconstructed_dbn_no_noise_binarized = mlp_test_model.predict(
            binarize_images(reconstructed_dbn_no_noise).reshape(1, -1))


        # add mse error below the plot
        plt.text(-105, 50, f"RBM: {np.argmax(prediction_reconstructed_rbm)}", fontsize=8, color='black')

        plt.text(-70, 50, f"Binarized RBM: {np.argmax(prediction_reconstructed_rbm_binarized)}", fontsize=8,
                 color='black')

        plt.text(-32, 50, f"DBN: {np.argmax(prediction_reconstructed_dbn)}", fontsize=8, color='black')

        plt.text(0, 50, f"Binarized DBN: {np.argmax(prediction_reconstructed_dbn_binarized)}", fontsize=8,
                 color='black')

        plt.show()