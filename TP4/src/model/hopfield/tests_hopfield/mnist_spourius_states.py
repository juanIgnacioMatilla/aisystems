import numpy as np
from sklearn.datasets import fetch_openml

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from TP4.hopfield_tests.main_hopfield import add_noise
from TP4.hopfield_tests.minst_hopfield import get_indexes_by_digit
from TP4.src.model.hopfield.hopfield_network import HopfieldNetwork
from TP4.src.model.hopfield.tests_hopfield.minist_metrics_differents_amount_no_pca import binarize

if __name__ == "__main__":
    # Fijar semilla para reproducibilidad
    np.random.seed(42)

    # Definir número de runs
    n_runs = 10  # Puedes ajustar este número según tus necesidades

    # Cargar el dataset MNIST
    print("Cargando el dataset MNIST...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    data = mnist.data
    targets = mnist.target.astype(int)

    # Estandarizar los datos
    print("Estandarizando los datos...")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Aplicar PCA para reducir la dimensionalidad
    print("Aplicando PCA para retener el 95% de la varianza...")
    pca_full = PCA().fit(data_scaled)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"Número de componentes PCA para retener el 95% de la varianza: {n_components}")
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data_scaled)

    # Binarizar las imágenes con umbral 0
    print("Binarizando los patrones...")
    patterns_pca = binarize(data_pca, threshold=0).astype(int)
    patterns_pca[patterns_pca == 0] = -1

    # Seleccionar un patrón por dígito (0-9)
    print("Seleccionando un patrón por dígito...")
    indices = []
    amount_of_digits_in_patterns = 1
    for digit in range(10):
        aux_indexes = get_indexes_by_digit(digit, targets)[:amount_of_digits_in_patterns]
        if not aux_indexes:
            raise ValueError(f"No se encontraron patrones para el dígito {digit}.")
        indices.extend(aux_indexes)
    selected_patterns = patterns_pca[indices]

    # Crear la red de Hopfield
    print("Inicializando la red de Hopfield...")
    hopfield_net = HopfieldNetwork(selected_patterns)

    # Definir niveles de ruido a evaluar
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5,0.55,0.6]
    correct_counts = {noise: [] for noise in noise_levels}  # Inicializar contador de estados correctos
    incorrect_counts = {noise: [] for noise in noise_levels}  # Inicializar contador de estados incorrectos
    spurious_counts = {noise: [] for noise in noise_levels}  # Inicializar contador de estados espurios

    # Iterar sobre cada run
    print(f"Realizando {n_runs} runs para cada nivel de ruido...")
    for run in range(n_runs):
        print(f"Run {run + 1}/{n_runs}")
        for test_number, pattern in enumerate(selected_patterns):
            for noise_level in noise_levels:
                # Añadir ruido al patrón
                noisy_pattern = add_noise(pattern, noise_level=noise_level)

                # Recuperar el patrón usando la red de Hopfield
                recovered_pattern, _, _ = hopfield_net.get_similar(noisy_pattern.copy(), max_iters=100)

                # Verificar si el patrón recuperado es correcto, incorrecto o espurio
                if np.array_equal(recovered_pattern, pattern):
                    # Estado Correcto
                    correct_counts[noise_level].append(1)
                    incorrect_counts[noise_level].append(0)
                    spurious_counts[noise_level].append(0)
                elif any(np.array_equal(recovered_pattern, stored_pattern) for stored_pattern in selected_patterns):
                    # Estado Incorrecto (coincide con otro patrón almacenado)
                    correct_counts[noise_level].append(0)
                    incorrect_counts[noise_level].append(1)
                    spurious_counts[noise_level].append(0)
                else:
                    # Estado Espurio
                    correct_counts[noise_level].append(0)
                    incorrect_counts[noise_level].append(0)
                    spurious_counts[noise_level].append(1)

    # Calcular estadísticas (media y desviación estándar) para cada nivel de ruido
    print("Calculando estadísticas de estados correctos, incorrectos y espurios...")
    correct_stats = {
        noise: {
            'mean': np.mean(counts),
            'std': np.std(counts)
        }
        for noise, counts in correct_counts.items()
    }

    incorrect_stats = {
        noise: {
            'mean': np.mean(counts),
            'std': np.std(counts)
        }
        for noise, counts in incorrect_counts.items()
    }

    spurious_stats = {
        noise: {
            'mean': np.mean(counts),
            'std': np.std(counts)
        }
        for noise, counts in spurious_counts.items()
    }

    # Graficar la cantidad de estados correctos, incorrectos y espurios con gráficos de dispersión y barras de error
    print("Graficando la cantidad de estados correctos, incorrectos y espurios...")

    # Preparar datos para graficar
    noise_levels_sorted = sorted(noise_levels)
    correct_means = [correct_stats[noise]['mean'] for noise in noise_levels_sorted]
    correct_stds = [correct_stats[noise]['std'] for noise in noise_levels_sorted]

    incorrect_means = [incorrect_stats[noise]['mean'] for noise in noise_levels_sorted]
    incorrect_stds = [incorrect_stats[noise]['std'] for noise in noise_levels_sorted]

    spurious_means = [spurious_stats[noise]['mean'] for noise in noise_levels_sorted]
    spurious_stds = [spurious_stats[noise]['std'] for noise in noise_levels_sorted]

    # Graficar Estados Correctos
    plt.figure(figsize=(12, 8))
    plt.errorbar(
        noise_levels_sorted,
        correct_means,
        yerr=correct_stds,
        fmt='o-',  # 'o' para puntos, '-' para líneas conectando
        ecolor='gray',
        capsize=5,
        capthick=1,
        markerfacecolor='green',
        markersize=8,
        linewidth=2,
        label='Estados Correctos'
    )
    plt.xlabel('Nivel de Ruido', fontsize=16)
    plt.ylabel('Cantidad de Estados Correctos', fontsize=16)
    plt.title('Cantidad de Estados Correctos para Diferentes Niveles de Ruido', fontsize=18)
    plt.xticks(noise_levels_sorted, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show()
    print("Graficación de los estados correctos completada.")

    # Graficar Estados Incorrectos
    plt.figure(figsize=(12, 8))
    plt.errorbar(
        noise_levels_sorted,
        incorrect_means,
        yerr=incorrect_stds,
        fmt='s--',  # 's' para cuadrados, '--' para líneas discontinuas
        ecolor='gray',
        capsize=5,
        capthick=1,
        markerfacecolor='orange',
        markersize=8,
        linewidth=2,
        label='Estados Incorrectos'
    )
    plt.xlabel('Nivel de Ruido', fontsize=16)
    plt.ylabel('Cantidad de Estados Incorrectos', fontsize=16)
    plt.title('Cantidad de Estados Incorrectos para Diferentes Niveles de Ruido', fontsize=18)
    plt.xticks(noise_levels_sorted, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show()
    print("Graficación de los estados incorrectos completada.")

    # Graficar Estados Espurios
    plt.figure(figsize=(12, 8))
    plt.errorbar(
        noise_levels_sorted,
        spurious_means,
        yerr=spurious_stds,
        fmt='d-.',  # 'd' para diamantes, '-.' para líneas de puntos y rayas
        ecolor='gray',
        capsize=5,
        capthick=1,
        markerfacecolor='red',
        markersize=8,
        linewidth=2,
        label='Estados Espurios'
    )
    plt.xlabel('Nivel de Ruido', fontsize=16)
    plt.ylabel('Cantidad de Estados Espurios', fontsize=16)
    plt.title('Cantidad de Estados Espurios para Diferentes Niveles de Ruido', fontsize=18)
    plt.xticks(noise_levels_sorted, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show()
    print("Graficación de los estados espurios completada.")