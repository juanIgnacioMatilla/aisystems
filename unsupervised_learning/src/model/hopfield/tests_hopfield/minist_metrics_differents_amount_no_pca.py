import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import random
import time


# Definición de la función de signo
def sgn(x):
    return 1 if x >= 0 else -1


# Definición de la red de Hopfield
class HopfieldNetwork:
    def __init__(self, patterns):
        self.n_neurons = patterns.shape[1]  # Número de neuronas
        self.weights = self._initialize_weights(patterns)

    def _initialize_weights(self, patterns):
        """Inicializa los pesos usando el aprendizaje Hebbiano."""
        weights = np.zeros((self.n_neurons, self.n_neurons))
        for p in patterns:
            weights += np.outer(p, p)
        weights /= self.n_neurons
        np.fill_diagonal(weights, 0)  # Sin auto-conexiones
        return weights

    def update(self, state):
        """Actualiza asincrónicamente el estado de la red."""
        new_state = state.copy()
        indices = list(range(self.n_neurons))
        random.shuffle(indices)  # Orden aleatorio para la actualización
        for i in indices:
            h = np.dot(self.weights[i], new_state)
            new_state[i] = sgn(h)
        return new_state

    def run(self, initial_state, max_iters=100):
        """Ejecuta la red hasta la convergencia o el número máximo de iteraciones."""
        state = initial_state.copy()
        states_history = [state.copy()]
        energy_history = [self.energy(state)]
        for _ in range(max_iters):
            new_state = self.update(state)
            if np.array_equal(new_state, state):
                break
            states_history.append(new_state.copy())
            energy_history.append(self.energy(new_state))
            state = new_state
        return state, states_history, energy_history

    def energy(self, state):
        """Calcula la energía actual de la red."""
        return -0.5 * np.dot(state, np.dot(self.weights, state))


# Función para binarizar los datos
def binarize(data, threshold=0):
    """Convierte los datos a valores binarios (-1, +1) basados en un umbral."""
    binary = np.where(data > threshold, 1, -1)
    return binary


# Función para añadir ruido a un patrón
def add_noise(pattern, noise_level):
    """Invierte una proporción de bits en el patrón basado en el nivel de ruido."""
    noisy = pattern.copy()
    n_flips = int(noise_level * len(pattern))
    if n_flips == 0:
        return noisy
    flip_indices = np.random.choice(len(pattern), n_flips, replace=False)
    noisy[flip_indices] *= -1
    return noisy


def main():
    # Parámetros
    runs_per_k = 10  # Número de ejecuciones por cantidad de patrones almacenados
    max_k = 50  # Máximo número de patrones a almacenar
    step_k = 5  # Incremento en la cantidad de patrones
    noise_level = 0.05  # Nivel de ruido fijo para esta comparación

    # Cargar el dataset MNIST
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    data = mnist['data']
    targets = mnist['target'].astype(int)
    # Binarizar los datos
    patterns_pca = binarize(data, threshold=127).astype(int)
    patterns_pca[patterns_pca == 0] = -1

    # Seleccionar índices de patrones para asegurar diversidad
    unique_digits = np.unique(targets)
    digit_to_indices = {digit: np.where(targets == digit)[0] for digit in unique_digits}

    # Definir el rango de K (número de patrones almacenados)
    k_values = list(range(1, max_k + 1, step_k))
    precisions = []
    precisions_std = []

    # Evaluar para cada K
    print("Evaluando la precisión para diferentes cantidades de patrones almacenados...")
    for K in k_values:
        print(f"\nCantidad de Patrones Almacenados (K): {K}")
        precision_runs = []
        start_time = time.time()
        for run in range(runs_per_k):
            # Seleccionar K patrones al azar, asegurando que sean de diferentes dígitos si K <= 10
            selected_patterns = []
            if K <= len(unique_digits):
                # Seleccionar un patrón aleatorio de cada dígito hasta K
                digits = np.random.choice(unique_digits, K, replace=False)
                for digit in digits:
                    indices = digit_to_indices[digit]
                    selected_index = np.random.choice(indices)
                    selected_patterns.append(patterns_pca[selected_index])
            else:
                # Si K > número de dígitos, permitir múltiples patrones por dígito
                for _ in range(K):
                    digit = np.random.choice(unique_digits)
                    indices = digit_to_indices[digit]
                    selected_index = np.random.choice(indices)
                    selected_patterns.append(patterns_pca[selected_index])
            selected_patterns = np.array(selected_patterns)

            # Inicializar la red de Hopfield con K patrones
            hopfield_net = HopfieldNetwork(selected_patterns)

            # Evaluar la precisión
            correct = 0
            for i, original_pattern in enumerate(selected_patterns):
                # Añadir ruido al patrón original
                noisy_pattern = add_noise(original_pattern, noise_level)

                # Ejecutar la red de Hopfield
                recovered_pattern, _, _ = hopfield_net.run(noisy_pattern, max_iters=100)

                # Verificar si el patrón recuperado coincide con el original
                if np.array_equal(recovered_pattern, original_pattern):
                    correct += 1

            # Calcular la precisión para esta ejecución
            precision = correct / K
            precision_runs.append(precision)

            print(f"  Run {run + 1}/{runs_per_k}: Precisión = {precision * 100:.1f}%")

        # Calcular media y desviación estándar para K
        mean_precision = np.mean(precision_runs)
        std_precision = np.std(precision_runs)
        precisions.append(mean_precision)
        precisions_std.append(std_precision)

        elapsed_time = time.time() - start_time
        print(
            f"  Precisión Media: {mean_precision * 100:.1f}% ± {std_precision * 100:.1f}% (Tiempo: {elapsed_time:.2f}s)")

    # Graficar Precisión vs Número de Patrones Almacenados
    plt.figure(figsize=(12, 6))
    plt.errorbar(k_values, precisions, yerr=precisions_std, fmt='-o', color='blue',
                 ecolor='lightblue', capsize=5, label='Precisión')
    plt.xlabel('Número de Patrones Almacenados (K)', fontsize=12)
    plt.ylabel('Precisión', fontsize=12)
    plt.title('Precisión vs. Número de Patrones Almacenados en la Red de Hopfield (Sin PCA)', fontsize=14)
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Opcional: Visualización de ejemplos específicos con un nivel de ruido seleccionado
    example_k = min(k_values, key=lambda x: abs(x - 10))  # Seleccionar K cercano a 10
    example_noise_level = noise_level  # 20% de ruido
    print(f"\nVisualizando ejemplos de recuperación con K={example_k} y {example_noise_level * 100:.0f}% de ruido...")

    # Seleccionar un conjunto de patrones para visualizar
    selected_patterns = []
    digits = np.random.choice(unique_digits, example_k, replace=False) if example_k <= len(unique_digits) else None
    if example_k <= len(unique_digits):
        for digit in digits:
            indices = digit_to_indices[digit]
            selected_index = np.random.choice(indices)
            selected_patterns.append(patterns_pca[selected_index])
    else:
        for _ in range(example_k):
            digit = np.random.choice(unique_digits)
            indices = digit_to_indices[digit]
            selected_index = np.random.choice(indices)
            selected_patterns.append(patterns_pca[selected_index])
    selected_patterns = np.array(selected_patterns)

    # Inicializar la red de Hopfield con K patrones
    hopfield_net = HopfieldNetwork(selected_patterns)

    # Visualizar algunos ejemplos
    num_examples = min(5, example_k)  # Mostrar hasta 5 ejemplos
    plt.figure(figsize=(12, 9))
    for i in range(num_examples):
        original_pattern = selected_patterns[i]
        noisy_pattern = add_noise(original_pattern, example_noise_level)
        recovered_pattern, _, _ = hopfield_net.run(noisy_pattern, max_iters=100)

        # Convertir patrones binarizados a imágenes
        original_image = original_pattern.reshape(28, 28)
        noisy_image = noisy_pattern.reshape(28, 28)
        recovered_image = recovered_pattern.reshape(28, 28)

        # Visualizar patrón original
        plt.subplot(num_examples, 3, 3 * i + 1)
        plt.imshow(original_image, cmap='gray')
        plt.title('Original')
        plt.axis('off')

        # Visualizar patrón con ruido
        plt.subplot(num_examples, 3, 3 * i + 2)
        plt.imshow(noisy_image, cmap='gray')
        plt.title(f'Con Ruido ({int(noise_level * 100)}%)')
        plt.axis('off')

        # Visualizar patrón recuperado
        plt.subplot(num_examples, 3, 3 * i + 3)
        plt.imshow(recovered_image, cmap='gray')
        plt.title('Recuperado')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
