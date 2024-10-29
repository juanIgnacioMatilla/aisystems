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
    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    energy_histories = {noise: [] for noise in noise_levels}

    # Iterar sobre cada patrón seleccionado
    print("Procesando cada patrón con diferentes niveles de ruido...")
    for test_number, pattern in enumerate(selected_patterns):
        for noise_level in noise_levels:
            # Añadir ruido al patrón
            noisy_pattern = add_noise(pattern, noise_level=noise_level)

            # Recuperar el patrón usando la red de Hopfield
            _, _, energy_history = hopfield_net.get_similar(noisy_pattern.copy(), max_iters=100)

            # Almacenar el historial de energía
            energy_histories[noise_level].append(energy_history)

    # Procesar los historiales de energía para graficar
    print("Procesando historiales de energía para graficar...")
    avg_energy_histories = {}
    std_energy_histories = {}
    for noise_level, histories in energy_histories.items():
        # Encontrar la longitud máxima de las secuencias de energía
        max_length = max(len(hist) for hist in histories)

        # Rellenar los historiales más cortos con su último valor para igualar longitudes
        padded_histories = [hist + [hist[-1]] * (max_length - len(hist)) for hist in histories]

        # Calcular el promedio y el desvío estándar de la energía en cada iteración
        avg_energy = np.mean(padded_histories, axis=0)
        std_energy = np.std(padded_histories, axis=0)
        avg_energy_histories[noise_level] = avg_energy
        std_energy_histories[noise_level] = std_energy

    # Graficar los historiales de energía con barras de error (desvío estándar)
    print("Graficando la evolución de la energía...")

    # Configurar parámetros de estilo globales para mayor legibilidad
    plt.rcParams.update({
        'font.size': 14,  # Tamaño de fuente general
        'axes.labelsize': 16,  # Tamaño de las etiquetas de los ejes
        'axes.titlesize': 18,  # Tamaño del título del gráfico
        'legend.fontsize': 12,  # Tamaño de la leyenda
        'xtick.labelsize': 12,  # Tamaño de las marcas del eje x
        'ytick.labelsize': 12,  # Tamaño de las marcas del eje y
        'lines.linewidth': 2,  # Grosor de las líneas
        'figure.figsize': (10, 7)  # Tamaño de la figura (ajustable)
    })

    plt.figure()  # La figura ya está configurada en rcParams

    for noise_level, avg_energy in avg_energy_histories.items():
        std_energy = std_energy_histories[noise_level]
        plt.errorbar(range(len(avg_energy)), avg_energy, yerr=std_energy, label=f'Ruido {noise_level}', capsize=3)

    plt.xlabel('Iteraciones')
    plt.ylabel('Energía')
    plt.title('Evolución de la Energía hasta la Convergencia para Diferentes Niveles de Ruido')
    plt.legend(title='Niveles de Ruido')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print("Graficación completada.")
