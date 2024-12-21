import json

from matplotlib import pyplot as plt

from TP4.src.model.boltzman.boltzmann_utils import load_mnist_data_split, add_noise_to_image
from TP4.src.model.boltzman.deep_belief_network import DBN


def load_config(config_path):
    """
    Carga la configuración desde un archivo JSON.

    Args:
        config_path (str): Ruta al archivo de configuración.

    Returns:
        dict: Diccionario con la configuración cargada.
    """
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config


def main(config_path='config.json'):
    # Cargar la configuración
    config = load_config(config_path)

    # Extraer parámetros de la configuración
    dataset = 'mnist'
    layer_sizes = config['model']['layer_sizes']

    train_config = config['training']
    epochs = train_config['epochs']
    batch_size = train_config['batch_size']
    learning_rate = train_config['learning_rate']
    k = train_config['k']

    noise_level = config['noise']['noise_level']

    num_images = config['visualization']['num_images']
    x_train, y_train, x_test, y_test = load_mnist_data_split()

    n_samples, n_visible = x_train.shape

    # Inicializar la DBN
    dbn = DBN(layer_sizes)

    # Preentrenar la DBN
    dbn.pretrain(x_train, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, k=k)

    for i in range(num_images):
        # Seleccionar una imagen de prueba
        test_image = x_train[i]

        # Agregar ruido a la imagen de prueba
        noisy_image = add_noise_to_image(test_image, noise_level=noise_level)

        # Reconstruir la imagen a través de la DBN
        reconstructed_image = dbn.reconstruct(noisy_image)

        # Visualizar los resultados
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.title("Imagen Original")
        plt.imshow(test_image.reshape(28, 28), cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title(f"Imagen con Ruido ({int(noise_level * 100)}%)")
        plt.imshow(noisy_image.reshape(28, 28), cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Imagen Reconstruida DBN")
        plt.imshow(reconstructed_image.reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.suptitle("DBN", fontsize=16)
        plt.show()


if __name__ == "__main__":
    main()
