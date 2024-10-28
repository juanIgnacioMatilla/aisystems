import time

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from TP4.src.model.hopfield.tests_hopfield.minist_metrics_differents_amount_no_pca import binarize, HopfieldNetwork, \
    add_noise


def main():
    noise_level = 0.2    # Nivel de ruido fijo para esta comparación

    # Cargar el dataset MNIST
    print("Cargando el dataset MNIST...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    data = mnist['data']
    targets = mnist['target'].astype(int)

    # Estandarizar los datos
    print("Estandarizando los datos...")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Aplicar PCA para conservar el 95% de la varianza
    print("Aplicando PCA para reducir la dimensionalidad...")
    pca_full = PCA().fit(data_scaled)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"Número de componentes PCA para conservar el 95% de la varianza: {n_components}")
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data_scaled)

    # Binarizar los datos transformados por PCA
    print("Binarizando los datos transformados por PCA...")
    patterns_pca = binarize(data_pca, threshold=0).astype(int)
    patterns_pca[patterns_pca == 0] = -1

    # Seleccionar índices de patrones para asegurar diversidad
    unique_digits = np.unique(targets)
    digit_to_indices = {digit: np.where(targets == digit)[0] for digit in unique_digits}

    # Visualización de ejemplos específicos con un nivel de ruido seleccionado
    example_k = 10  # Puedes ajustar este valor según tus necesidades
    example_noise_level = noise_level  # 20% de ruido
    print(f"\nVisualizando ejemplos de recuperación con K={example_k} y {example_noise_level*100:.0f}% de ruido...")

    # Seleccionar un conjunto de patrones para visualizar
    selected_patterns = []
    if example_k <= len(unique_digits):
        # Seleccionar K dígitos distintos
        digits = np.random.choice(unique_digits, example_k, replace=False)
        for digit in digits:
            indices = digit_to_indices[digit]
            selected_index = np.random.choice(indices)
            selected_patterns.append(patterns_pca[selected_index])
    else:
        # Seleccionar K patrones aleatorios
        for _ in range(example_k):
            digit = np.random.choice(unique_digits)
            indices = digit_to_indices[digit]
            selected_index = np.random.choice(indices)
            selected_patterns.append(patterns_pca[selected_index])
    selected_patterns = np.array(selected_patterns)

    # Inicializar la red de Hopfield con K patrones
    hopfield_net = HopfieldNetwork(selected_patterns)

    # Visualización de ejemplos específicos con un nivel de ruido seleccionado
    example_k = 10  # Puedes ajustar este valor según tus necesidades
    example_noise_level = noise_level  # 20% de ruido
    print(f"\nVisualizando ejemplos de recuperación con K={example_k} y {example_noise_level * 100:.0f}% de ruido...")

    # Seleccionar un conjunto de patrones para visualizar
    selected_patterns = []
    selected_original_indices = []
    if example_k <= len(unique_digits):
        # Seleccionar K dígitos distintos
        digits = np.random.choice(unique_digits, example_k, replace=False)
        for digit in digits:
            indices = digit_to_indices[digit]
            selected_index = np.random.choice(indices)
            selected_patterns.append(patterns_pca[selected_index])
            selected_original_indices.append(selected_index)
    else:
        # Seleccionar K patrones aleatorios
        for _ in range(example_k):
            digit = np.random.choice(unique_digits)
            indices = digit_to_indices[digit]
            selected_index = np.random.choice(indices)
            selected_patterns.append(patterns_pca[selected_index])
            selected_original_indices.append(selected_index)
    selected_patterns = np.array(selected_patterns)

    # Inicializar la red de Hopfield con K patrones
    hopfield_net = HopfieldNetwork(selected_patterns)

    # Visualizar algunos ejemplos
    num_examples = min(5, example_k)  # Mostrar hasta 5 ejemplos
    plt.figure(figsize=(15, 10))
    for i in range(num_examples):
        original_pca_pattern = selected_patterns[i]
        original_index = selected_original_indices[i]
        noisy_pattern = add_noise(original_pca_pattern, example_noise_level)
        recovered_pattern, _, _ = hopfield_net.run(noisy_pattern, max_iters=100)

        # Obtener la imagen original antes de PCA y estandarización
        original_image_raw = data[original_index].reshape(28, 28)

        # Obtener la imagen original después de PCA y binarización
        original_image_pca = pca.inverse_transform(original_pca_pattern.reshape(1, -1))
        original_image_pca = scaler.inverse_transform(original_image_pca).reshape(28, 28)
        original_image_pca = np.clip(original_image_pca, 0, 255)

        # Obtener la imagen con ruido después de PCA y binarización
        noisy_image_pca = pca.inverse_transform(noisy_pattern.reshape(1, -1))
        noisy_image_pca = scaler.inverse_transform(noisy_image_pca).reshape(28, 28)
        noisy_image_pca = np.clip(noisy_image_pca, 0, 255)

        # Obtener la imagen recuperada después de PCA y binarización
        recovered_image_pca = pca.inverse_transform(recovered_pattern.reshape(1, -1))
        recovered_image_pca = scaler.inverse_transform(recovered_image_pca).reshape(28, 28)
        recovered_image_pca = np.clip(recovered_image_pca, 0, 255)

        # Visualizar patrón original (antes de PCA)
        plt.subplot(num_examples, 4, 4 * i + 1)
        plt.imshow(original_image_raw, cmap='gray')
        plt.title('Original (Sin PCA)')
        plt.axis('off')

        # Visualizar patrón original (Después de PCA y Binarización)
        plt.subplot(num_examples, 4, 4 * i + 2)
        plt.imshow(original_image_pca, cmap='gray')
        plt.title('Original post PCA')
        plt.axis('off')

        # Visualizar patrón con ruido
        plt.subplot(num_examples, 4, 4 * i + 3)
        plt.imshow(noisy_image_pca, cmap='gray')
        plt.title(f'Con Ruido ({int(noise_level * 100)}%)')
        plt.axis('off')

        # Visualizar patrón recuperado
        plt.subplot(num_examples, 4, 4 * i + 4)
        plt.imshow(recovered_image_pca, cmap='gray')
        plt.title('Recuperado')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
