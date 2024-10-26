import time

import numpy as np
import matplotlib.pyplot as plt

from TP4.src.model.boltzman.boltzmann_utils import load_mnist_data_split, add_noise_to_image


class RBM:
    def __init__(self, n_visible, n_hidden):
        self.n_visible = n_visible  # Número de unidades visibles (784 para MNIST)
        self.n_hidden = n_hidden  # Número de unidades ocultas
        self.training_time = 0  # Para rastrear el tiempo total de entrenamiento
        # Inicializar pesos y sesgos
        np.random.seed(42)
        self.weights = np.random.normal(0, 0.01, (self.n_visible, self.n_hidden))
        self.visible_bias = np.zeros(self.n_visible)
        self.hidden_bias = np.zeros(self.n_hidden)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sample_hidden(self, visible):
        """
        Muestra las unidades ocultas dadas las visibles.

        :param visible: Array de forma (batch_size, n_visible)
        :return: Tuple de (probabilidades ocultas, muestras binarias ocultas)
        """
        activation = np.dot(visible, self.weights) + self.hidden_bias
        prob_hidden = self.sigmoid(activation)
        hidden_sample = (prob_hidden > np.random.rand(*prob_hidden.shape)).astype(np.float32)
        return prob_hidden, hidden_sample

    def sample_visible(self, hidden):
        """
        Muestra las unidades visibles dadas las ocultas.

        :param hidden: Array de forma (batch_size, n_hidden)
        :return: Tuple de (probabilidades visibles, muestras binarias visibles)
        """
        activation = np.dot(hidden, self.weights.T) + self.visible_bias
        prob_visible = self.sigmoid(activation)
        visible_sample = (prob_visible > np.random.rand(*prob_visible.shape)).astype(np.float32)
        return prob_visible, visible_sample

    def contrastive_divergence(self, data, learning_rate=0.1, k=1):
        """
        Realiza el algoritmo de divergencia contrastiva para actualizar los pesos y sesgos.

        :param data: Array de forma (batch_size, n_visible)
        :param learning_rate: Tasa de aprendizaje
        :param k: Número de pasos de Gibbs Sampling
        """
        # Paso positivo: Muestreo de unidades ocultas dadas las visibles
        prob_hidden_pos, hidden_sample = self.sample_hidden(data)
        positive_grad = np.dot(data.T, prob_hidden_pos)

        # Inicializar muestras para la cadena de Gibbs
        visible_sample = data.copy()
        hidden_sample = hidden_sample.copy()

        # Paso negativo: Ejecutar k pasos de Gibbs Sampling
        for _ in range(k):
            prob_visible_neg, visible_sample = self.sample_visible(hidden_sample)
            prob_hidden_neg, hidden_sample = self.sample_hidden(visible_sample)

        negative_grad = np.dot(visible_sample.T, prob_hidden_neg)

        # Actualizar pesos y sesgos
        self.weights += learning_rate * (positive_grad - negative_grad) / data.shape[0]
        self.visible_bias += learning_rate * np.mean(data - visible_sample, axis=0)
        self.hidden_bias += learning_rate * np.mean(prob_hidden_pos - prob_hidden_neg, axis=0)

    def train(self, data, epochs=10, batch_size=100, learning_rate=0.1, k=1):
        """
        Entrena la RBM utilizando el algoritmo de contraste divergente.

        :param data: Array de datos de entrenamiento de forma (n_samples, n_visible)
        :param epochs: Número de épocas de entrenamiento
        :param batch_size: Tamaño de cada lote
        :param learning_rate: Tasa de aprendizaje
        :param k: Número de pasos de Gibbs Sampling
        """
        time_start = time.time()
        n_samples = data.shape[0]
        for epoch in range(epochs):
            # Barajar los datos
            np.random.shuffle(data)
            for i in range(0, n_samples, batch_size):
                batch = data[i:i + batch_size]
                self.contrastive_divergence(batch, learning_rate, k)
            print(f"RBM: Época {epoch + 1}/{epochs} completada")

        self.training_time = time.time() - time_start

    def reconstruct(self, data):
        """
        Reconstruye las visibles a partir de las visibles.

        :param data: Array de forma (batch_size, n_visible)
        :return: Array reconstruido de forma (batch_size, n_visible)
        """
        prob_hidden, _ = self.sample_hidden(data)
        prob_visible, _ = self.sample_visible(prob_hidden)
        return prob_visible

    def reconstruct_visible(self, hidden):
        """
        Reconstruye las visibles a partir de las ocultas.

        :param hidden: Array de forma (batch_size, n_hidden)
        :return: Array reconstruido de forma (batch_size, n_visible)
        """
        prob_visible, _ = self.sample_visible(hidden)
        return prob_visible

    def transform(self, data):
        """
        Obtiene las activaciones ocultas dadas las visibles.

        :param data: Array de forma (batch_size, n_visible)
        :return: Array de activaciones ocultas de forma (batch_size, n_hidden)
        """
        prob_hidden, _ = self.sample_hidden(data)
        return prob_hidden


if __name__ == "__main__":
    # Cargar los datos
    x_train = load_mnist_data_split()
    n_samples, n_visible = x_train.shape
    n_hidden = 64  # Puedes ajustar este valor según tus necesidades

    # Inicializar la RBM
    rbm = RBM(n_visible, n_hidden)

    # Entrenar la RBM
    rbm.train(x_train, epochs=5, batch_size=10, learning_rate=0.01, k=1)

    # Definir el nivel de ruido
    noise_level = 0.07

    for i in range(15):
        # Seleccionar una imagen de prueba
        test_image = x_train[i]

        # Agregar ruido a la imagen de prueba
        noisy_image = add_noise_to_image(test_image, noise_level=noise_level)

        # Reconstruir la imagen a partir de la imagen con ruido
        reconstructed_image = rbm.reconstruct(noisy_image)

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
        plt.title("Imagen Reconstruida")
        plt.imshow(reconstructed_image.reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.legend("RBM")
        plt.show()