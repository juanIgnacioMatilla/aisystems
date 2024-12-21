import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, binarize

from TP4.hopfield_tests.main_hopfield import add_noise
from TP4.src.model.hopfield.hopfield_network import HopfieldNetwork


def main():
    # Parámetros
    runs_per_noise_level = 50  # Número de ejecuciones por nivel de ruido
    noise_levels = np.linspace(0, 0.5, 11)  # 0%, 5%, ..., 50%

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
    print("Binarizando los datos...")
    patterns_pca = binarize(data_pca, threshold=0).astype(int)
    patterns_pca[patterns_pca == 0] = -1

    # Seleccionar patrones para entrenar la red de Hopfield
    print("Seleccionando patrones para entrenar la red de Hopfield...")
    selected_patterns = []
    for digit in range(10):
        indices = np.where(targets == digit)[0]
        if len(indices) < 1:
            continue
        amount_of_patterns_per_digit = 2
        for i in range(amount_of_patterns_per_digit):
            selected_patterns.append(patterns_pca[indices[i]])
    selected_patterns = np.array(selected_patterns)
    print(f"Número de patrones seleccionados: {selected_patterns.shape[0]}")

    # Inicializar la red de Hopfield
    print("Inicializando la red de Hopfield...")
    hopfield_net = HopfieldNetwork(selected_patterns)

    # Inicializar listas para almacenar métricas promedio y desviaciones estándar
    accuracies = np.zeros(len(noise_levels))
    accuracies_std = np.zeros(len(noise_levels))
    energies_final = np.zeros(len(noise_levels))
    energies_std = np.zeros(len(noise_levels))
    iterations_needed = np.zeros(len(noise_levels))
    iterations_std = np.zeros(len(noise_levels))

    # Evaluar la red en diferentes niveles de ruido con múltiples ejecuciones
    # print("Evaluando la red en diferentes niveles de ruido con múltiples ejecuciones...")
    # for idx, noise_level in enumerate(noise_levels):
    #     accuracy_runs = []
    #     energy_runs = []
    #     iterations_runs = []
    #     print(f"Nivel de Ruido: {noise_level*100:.0f}%")
    #     for run in range(runs_per_noise_level):
    #         correct = 0
    #         total_energy = 0
    #         total_iterations = 0
    #         for i, original_pattern in enumerate(selected_patterns):
    #             # Añadir ruido al patrón original
    #             noisy_pattern = add_noise(original_pattern, noise_level)
    #
    #             # Ejecutar la red de Hopfield
    #             recovered_pattern, states_history, energy_history = hopfield_net.get_similar(noisy_pattern, max_iters=100)
    #
    #             # Verificar si el patrón recuperado coincide con el original
    #             if np.array_equal(recovered_pattern, original_pattern):
    #                 correct += 1
    #
    #             total_energy += energy_history[-1]
    #             total_iterations += len(states_history)
    #
    #         # Calcular métricas para esta ejecución
    #         accuracy = correct / len(selected_patterns)
    #         mean_energy = np.mean(total_energy
    #         average_iterations = total_iterations / len(selected_patterns)
    #
    #         accuracy_runs.append(accuracy)
    #         energy_runs.append(average_energy)
    #         iterations_runs.append(average_iterations)
    #
    #     # Calcular media y desviación estándar para cada métrica
    #     accuracies[idx] = np.mean(accuracy_runs)
    #     accuracies_std[idx] = np.std(accuracy_runs)
    #     energies_final[idx] = np.mean(energy_runs)
    #     energies_std[idx] = np.std(energy_runs)
    #     iterations_needed[idx] = np.mean(iterations_runs)
    #     iterations_std[idx] = np.std(iterations_runs)
    #
    #     print(f"  Precisión Media: {accuracies[idx]*100:.1f}% ± {accuracies_std[idx]*100:.1f}%")
    #     print(f"  Energía Final Media: {energies_final[idx]:.2f} ± {energies_std[idx]:.2f}")
    #     print(f"  Iteraciones Promedio: {iterations_needed[idx]:.1f} ± {iterations_std[idx]:.1f}")
    #
    # # Graficar las métricas con barras de error
    # print("Generando gráficos de desempeño con barras de error...")
    # plt.figure(figsize=(18, 5))
    #
    # # Precisión vs Nivel de Ruido
    # plt.subplot(1, 3, 1)
    # plt.errorbar(noise_levels * 100, accuracies, yerr=accuracies_std, fmt='-o', color='blue', ecolor='lightblue', capsize=5)
    # plt.xlabel('Nivel de Ruido (%)')
    # plt.ylabel('Precisión')
    # plt.title('Precisión vs Nivel de Ruido')
    # plt.ylim(0, 1.05)
    # plt.grid(True)
    #
    # # Energía Final vs Nivel de Ruido
    # plt.subplot(1, 3, 2)
    # plt.errorbar(noise_levels * 100, energies_final, yerr=energies_std, fmt='-o', color='orange', ecolor='lightcoral', capsize=5)
    # plt.xlabel('Nivel de Ruido (%)')
    # plt.ylabel('Energía Final Promedio')
    # plt.title('Energía Final vs Nivel de Ruido')
    # plt.grid(True)
    #
    # # Iteraciones Necesarias vs Nivel de Ruido
    # plt.subplot(1, 3, 3)
    # plt.errorbar(noise_levels * 100, iterations_needed, yerr=iterations_std, fmt='-o', color='green', ecolor='lightgreen', capsize=5)
    # plt.xlabel('Nivel de Ruido (%)')
    # plt.ylabel('Iteraciones Promedio')
    # plt.title('Iteraciones vs Nivel de Ruido')
    # plt.grid(True)
    #
    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    main()