import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, binarize
from sklearn.datasets import fetch_openml
from TP4.hopfield_tests.main_hopfield import add_noise
from TP4.src.model.hopfield.hopfield_network import HopfieldNetwork


def get_indexes_by_digit(digit, targets):
    indexes = []
    for i, t in enumerate(targets):
        if t == digit:
            indexes.append(i)
    return indexes


if __name__ == "__main__":
    # Cargar el dataset MNIST
    mnist = fetch_openml('mnist_784', version=1)
    data = mnist.data
    targets = mnist.target.astype(int)

    # Estandarizar los datos
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    pca_full = PCA().fit(data_scaled)
    # Calcular cuantas compomentes representan el 95% de la varianza
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= 0.95) + 1
    # Aplicar PCA
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data_scaled)
    # Binarizar las imágenes con umbral 0
    patterns_pca = binarize(data_pca, threshold=0).astype(int)
    patterns_pca[patterns_pca == 0] = -1


    indices = []
    amount_of_digits_in_patterns = 1
    for i in range(10):
        aux_indexes = get_indexes_by_digit(i, targets)[:amount_of_digits_in_patterns]
        for aux_index in aux_indexes:
            indices.append(aux_index)
    selected_patterns = patterns_pca[indices]
    # Crear la red de Hopfield
    hopfield_net = HopfieldNetwork(selected_patterns)

    test_numbers = range(0, 9)
    for test_number in test_numbers:
        # Paso 0: Visualizar el patrón original para comparar
        original_index = indices[test_number]
        original_pattern = data.iloc[original_index]

        original_pattern_array = original_pattern.values
        plt.imshow(original_pattern_array.reshape(28, 28), cmap='gray')
        plt.title('Patrón Original')
        plt.show()

        # Visualizar el patrón almacenado en la red
        # Obtener el patrón almacenado (sin ruido)
        stored_pattern = selected_patterns[test_number]
        # Convertir a formato continuo
        stored_pattern_continuous = stored_pattern.astype(float)
        # Invertir la transformación PCA
        stored_pattern_original_space = pca.inverse_transform(stored_pattern_continuous)
        # Desestandarizar el patrón
        stored_pattern_original_space_2d = stored_pattern_original_space.reshape(1, -1)
        stored_pattern_descaled = scaler.inverse_transform(stored_pattern_original_space_2d)
        stored_pattern_descaled = stored_pattern_descaled.flatten()
        # Asegurar valores en el rango correcto
        stored_pattern_descaled = np.clip(stored_pattern_descaled, 0, 255)
        # Visualizar el patrón almacenado
        plt.imshow(stored_pattern_descaled.reshape(28, 28), cmap='gray')
        plt.title('Patrón Almacenado en la Red')
        plt.show()

        # Añadir ruido al patrón
        test_pattern = add_noise(selected_patterns[test_number], noise_level=0.2)

        # Visualizar el patrón con ruido
        # Convertir el patrón con ruido a formato continuo
        test_pattern_continuous = test_pattern.astype(float)
        # Invertir la transformación PCA
        test_pattern_original_space = pca.inverse_transform(test_pattern_continuous)
        # Desestandarizar el patrón reconstruido
        test_pattern_original_space_2d = test_pattern_original_space.reshape(1, -1)
        test_pattern_descaled = scaler.inverse_transform(test_pattern_original_space_2d)
        test_pattern_descaled = test_pattern_descaled.flatten()
        # Asegurar que los valores estén en el rango correcto
        test_pattern_descaled = np.clip(test_pattern_descaled, 0, 255)
        # Visualizar el patrón con ruido
        plt.imshow(test_pattern_descaled.reshape(28, 28), cmap='gray')
        plt.title('Patrón con Ruido')
        plt.show()

        # Recuperar el patrón
        recovered_pattern, _, _ = hopfield_net.get_similar(test_pattern.copy(), 100)
        # Convertir el patrón recuperado a formato continuo
        retrieved_pattern_continuous = recovered_pattern.astype(float)
        # Invertir la transformación PCA
        retrieved_pattern_original_space = pca.inverse_transform(retrieved_pattern_continuous)
        # Desestandarizar el patrón reconstruido
        retrieved_pattern_original_space_2d = retrieved_pattern_original_space.reshape(1, -1)
        retrieved_pattern_descaled = scaler.inverse_transform(retrieved_pattern_original_space_2d)
        retrieved_pattern_descaled = retrieved_pattern_descaled.flatten()
        # Visualizar el patrón recuperado
        plt.imshow(retrieved_pattern_descaled.reshape(28, 28), cmap='gray')
        plt.title('Patrón Recuperado Reconstruido')
        plt.show()
