import random

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml

from TP4.src.model.hopfield.orthogonality import patterns
from TP4.src.model.hopfield.tests_hopfield.compare_orthogonality_pca_nopca import binarize, \
    calculate_average_orthogonality


def main():
    # Parámetros
    K_mnist = 1000  # Número de patrones a seleccionar para MNIST en cada ejecución
    num_runs = 10  # Número de ejecuciones para calcular la media y desviación estándar
    # Cargar el dataset MNIST
    print("Cargando el dataset MNIST...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    data = mnist['data']

    # Binarizar los datos originales sin PCA
    print("Binarizando los datos originales sin PCA...")
    patterns_original = binarize(data, threshold=127).astype(int)
    patterns_original[patterns_original == 0] = -1

    # Preparar patrones de letras A-Z
    print("Preparando patrones de letras A-Z...")
    letters = list(patterns.keys())
    selected_patterns_letters = np.array([patterns[letter] for letter in letters])

    # Calcular la ortogonalidad promedio para las letras A-Z
    print("Calculando la ortogonalidad promedio para las letras A-Z...")
    avg_ortho_letters = calculate_average_orthogonality(selected_patterns_letters)

    # Inicializar lista para almacenar las ortogonalidades de MNIST
    ortho_mnist_runs = []

    # Realizar múltiples ejecuciones para MNIST
    print(f"Realizando {num_runs} ejecuciones seleccionando {K_mnist} patrones aleatorios cada vez para MNIST...")
    for run in range(num_runs):
        # Seleccionar K patrones aleatorios de MNIST sin reemplazo
        random_indices_mnist = random.sample(range(data.shape[0]), K_mnist)
        selected_patterns_mnist = patterns_original[random_indices_mnist]

        # Calcular la ortogonalidad promedio para esta ejecución
        avg_ortho_mnist = calculate_average_orthogonality(selected_patterns_mnist)
        ortho_mnist_runs.append(avg_ortho_mnist)

        print(f"  Run {run + 1}/{num_runs}: Ortogonalidad Promedio = {avg_ortho_mnist:.2f}")

    # Calcular la media y la desviación estándar de las ortogonalidades de MNIST
    mean_ortho_mnist = np.mean(ortho_mnist_runs)
    std_ortho_mnist = np.std(ortho_mnist_runs)
    print(std_ortho_mnist)

    # Mostrar los resultados
    print("\nComparación de la Ortogonalidad Promedio:")
    print(f"MNIST - K={K_mnist}: Media = {mean_ortho_mnist:.3f}, Desviación Estándar = {std_ortho_mnist:.2f}")
    print(f"Letras A-Z (5x5): Ortogonalidad Promedio = {avg_ortho_letters:.2f}")

    # Visualización de los resultados con barras de error para MNIST
    labels = ['MNIST', 'Letras A-Z (5x5)']
    averages = [mean_ortho_mnist, avg_ortho_letters]
    errors = [std_ortho_mnist, 0]  # No hay desviación estándar para letras A-Z
    colors = ['skyblue', 'salmon']

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, averages, yerr=errors, color=colors, capsize=10, alpha=0.7)
    plt.ylabel('Ortogonalidad Promedio')
    plt.title('Comparación de la Ortogonalidad Promedio entre MNIST y Letras A-Z')

    # Añadir etiquetas de valor encima de las barras
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if i == 0:
            plt.annotate(f'{height:.2f} ± {errors[i]:.3f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 5),  # 5 puntos de desplazamiento vertical
                         textcoords="offset points",
                         ha='center', va='bottom')
        else:
            plt.annotate(f'{height:.2f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 5),  # 5 puntos de desplazamiento vertical
                         textcoords="offset points",
                         ha='center', va='bottom')

    plt.ylim(0, max(averages) * 1.2)
    plt.show()


if __name__ == "__main__":
    main()
