from matplotlib import pyplot as plt
from TP4.src.model.boltzman.boltzmann_utils import load_mnist_data_split, add_noise_to_image, load_model
from TP4.src.model.boltzman.restricted_boltzmann_machine import RBM
import json
import matplotlib.pyplot as plt

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config


if __name__ == "__main__":
    # Cargar la configuración
    config = load_config('config.json')

    # Extraer parámetros de la configuración
    n_hidden = config['model']['n_hidden']
    n_visible = config['model'].get('n_visible', 784)  # Valor por defecto si no se especifica

    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    learning_rate = config['training']['learning_rate']
    k = config['training']['k']

    noise_level = config['noise']['noise_level']

    num_images = config['visualization']['num_images']

    # Cargar los datos
    x_train, y_train, x_test, y_test = load_mnist_data_split()
    n_samples_train, n_visible = x_train.shape
    n_samples_test, _ = x_test.shape

    # Inicializar la RBM
    rbm = RBM(n_visible, n_hidden)

    # Entrenar la RBM
    rbm.train(x_train, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, k=k)
    for i in range(num_images):
        # Seleccionar una imagen de prueba
        test_image = x_train[i]

        # Agregar ruido a la imagen de prueba
        noisy_image = add_noise_to_image(test_image, noise_level=noise_level)

        # Reconstruir la imagen a partir de la imagen con ruido
        reconstructed_image = rbm.reconstruct(noisy_image)

        # Visualizar los resultados
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.title("Imagen original binarizada")
        plt.imshow(test_image.reshape(28, 28), cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title(f"Imagen con ruido ({int(noise_level * 100)}%)")
        plt.imshow(noisy_image.reshape(28, 28), cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Imagen reconstruida")
        plt.imshow(reconstructed_image.reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.suptitle("RBM", fontsize=16)
        plt.show()