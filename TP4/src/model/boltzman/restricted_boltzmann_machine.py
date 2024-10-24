import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

class RBM:
    def __init__(self, n_visible, n_hidden):
        self.n_visible = n_visible  # Número de unidades visibles (784 para MNIST)
        self.n_hidden = n_hidden    # Número de unidades ocultas
        # Inicializar pesos y sesgos
        np.random.seed(42)
        self.weights = np.random.normal(0, 0.01, (self.n_visible, self.n_hidden))
        self.visible_bias = np.zeros(self.n_visible)
        self.hidden_bias = np.zeros(self.n_hidden)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sample_hidden(self, visible):
        activation = np.dot(visible, self.weights) + self.hidden_bias
        prob_hidden = self.sigmoid(activation)
        return prob_hidden, (prob_hidden > np.random.rand(len(prob_hidden))).astype(np.float32)

    def sample_visible(self, hidden):
        activation = np.dot(hidden, self.weights.T) + self.visible_bias
        prob_visible = self.sigmoid(activation)
        return prob_visible, (prob_visible > np.random.rand(len(prob_visible))).astype(np.float32)

    def contrastive_divergence(self, data, learning_rate=0.1, k=1):
        # Paso positivo: Muestreo de unidades ocultas dadas las visibles
        prob_hidden_pos, hidden_sample = self.sample_hidden(data)
        positive_grad = np.outer(data, prob_hidden_pos)

        # Inicializar muestras para la cadena de Gibbs
        visible_sample = data.copy()
        hidden_sample = hidden_sample.copy()

        # Paso negativo: Ejecutar k pasos de Gibbs Sampling
        for _ in range(k):
            prob_visible_neg, visible_sample = self.sample_visible(hidden_sample)
            prob_hidden_neg, hidden_sample = self.sample_hidden(visible_sample)

        negative_grad = np.outer(visible_sample, prob_hidden_neg)

        # Actualizar pesos y sesgos
        self.weights += learning_rate * (positive_grad - negative_grad)
        self.visible_bias += learning_rate * (data - visible_sample)
        self.hidden_bias += learning_rate * (prob_hidden_pos - prob_hidden_neg)

    def train(self, data, epochs=10, batch_size=100, learning_rate=0.1, k=1):
        n_samples = data.shape[0]
        for epoch in range(epochs):
            # Barajar los datos
            np.random.shuffle(data)
            for i in range(0, n_samples, batch_size):
                batch = data[i:i+batch_size]
                # Entrenar con cada muestra del lote
                for sample in batch:
                    self.contrastive_divergence(sample, learning_rate, k)
            print(f"Época {epoch+1}/{epochs} completada")

    def reconstruct(self, data):
        prob_hidden, _ = self.sample_hidden(data)
        prob_visible, _ = self.sample_visible(prob_hidden)
        return prob_visible

# Función para binarizar las imágenes
def binarize_images(images, threshold=127):
    return (images > threshold).astype(np.float32)

# Cargar y preprocesar los datos de MNIST
def load_mnist_data():
    (x_train, _), (_, _) = mnist.load_data()
    # Normalizar y binarizar las imágenes
    x_train = binarize_images(x_train)
    # Aplanar las imágenes
    x_train = x_train.reshape((x_train.shape[0], -1))
    return x_train

# Cargar y preprocesar los datos de MNIST
def load_mnist_data_no_binarized():
    (x_train, _), (_, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255
    x_train = x_train.reshape(60000, 784)
    return x_train


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


if __name__ == "__main__":
    # Cargar los datos
    x_train = load_mnist_data()
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

        plt.show()