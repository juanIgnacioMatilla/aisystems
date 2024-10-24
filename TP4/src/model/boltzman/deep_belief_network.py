import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from TP4.src.model.boltzman.restricted_boltzmann_machine import RBM, load_mnist_data
# Mostrar la original y la reconstruida
import matplotlib.pyplot as plt

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

def plot_images(original, reconstructed, n=1):
    plt.figure(figsize=(n * 2, 2))
    for i in range(n):
        # Imagen original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original.reshape(28, 28), cmap='gray')
        plt.title("Original")
        plt.axis('off')

        # Imagen reconstruida
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed.reshape(28, 28), cmap='gray')
        plt.title("Reconstruida")
        plt.axis('off')
    plt.show()


class DBN:
    def __init__(self, layer_sizes):
        """
        Inicializa la DBN con una lista de tamaños de capas.
        Por ejemplo, layer_sizes = [784, 500, 200, 50] crea una DBN con 3 capas RBM:
        - Primera RBM: 784 visibles, 500 ocultas
        - Segunda RBM: 500 visibles, 200 ocultas
        - Tercera RBM: 200 visibles, 50 ocultas
        """
        self.layer_sizes = layer_sizes
        self.rbms = []
        for i in range(len(layer_sizes) - 1):
            rbm = RBM(layer_sizes[i], layer_sizes[i + 1])
            self.rbms.append(rbm)

    def pretrain(self, data, epochs=10, batch_size=100, learning_rate=0.1, k=1):
        """
        Entrena cada RBM de la DBN de forma secuencial.

        :param data: Array de datos de entrenamiento de forma (n_samples, n_visible)
        :param epochs: Número de épocas de entrenamiento por RBM
        :param batch_size: Tamaño de cada lote
        :param learning_rate: Tasa de aprendizaje
        :param k: Número de pasos de Gibbs Sampling
        """
        input_data = data
        for idx, rbm in enumerate(self.rbms):
            print(f"\nEntrenando RBM {idx + 1}/{len(self.rbms)} con {rbm.n_visible} visibles y {rbm.n_hidden} ocultas")
            rbm.train(input_data, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, k=k)
            # Transformar los datos para la siguiente RBM
            input_data = rbm.transform(input_data)

    def reconstruct(self, data):
        """
        Reconstruye los datos pasando hacia adelante y luego hacia atrás por la DBN.

        :param data: Array de forma (batch_size, n_visible)
        :return: Array reconstruido de forma (batch_size, n_visible)
        """
        # Paso hacia adelante
        hidden = data
        for rbm in self.rbms:
            hidden = rbm.transform(hidden)
        # Paso hacia atrás
        for rbm in reversed(self.rbms):
            hidden = rbm.reconstruct_visible(hidden)
        return hidden

    def transform(self, data):
        """
        Obtiene las representaciones ocultas finales de la DBN.

        :param data: Array de forma (batch_size, n_visible)
        :return: Array de representaciones ocultas de forma (batch_size, n_hidden)
        """
        hidden = data
        for rbm in self.rbms:
            hidden = rbm.transform(hidden)
        return hidden


if __name__ == "__main__":
    # Cargar los datos
    x_train = load_mnist_data()
    n_samples, n_visible = x_train.shape

    # Definir la estructura de la DBN
    layer_sizes = [n_visible, 64, 10, 10]  # Puedes ajustar estos valores
    dbn = DBN(layer_sizes)

    # Preentrenar la DBN
    dbn.pretrain(x_train, epochs=10, batch_size=10, learning_rate=0.01, k=1)

    # Definir el nivel de ruido
    noise_level = 0.07

    for i in range(15):
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
        plt.title(f"Imagen con Ruido (Nivel {int(noise_level * 100)}%)")
        plt.imshow(noisy_image.reshape(28, 28), cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Imagen Reconstruida DBN")
        plt.imshow(reconstructed_image.reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.legend("DBN")
        plt.show()