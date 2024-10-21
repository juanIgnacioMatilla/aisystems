import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from TP4.main_hopfield import add_noise
from TP4.src.model.hopfield.hopfield_network import HopfieldNetwork
from TP4.src.model.hopfield.orthogonality import find_orthogonal_subset
from TP4.src.model.hopfield.tests_hopfield.orthogonality import categorize_result


def main():
    # Valores constantes de Min y Max Ortogonalidad
    min_orthogonality = 0
    max_orthogonality = 10

    # Obtener patrones con la ortogonalidad especificada
    patterns_set = find_orthogonal_subset(min_orthogonality, max_orthogonality)
    if patterns_set is None:
        print("No se encontraron patrones para los valores de ortogonalidad especificados.")
        exit()

    # Niveles de ruido a evaluar
    # De 0% a 100% en incrementos de 10%
    noise_levels = np.linspace(0, 1, 11)

    results_noise = []

    # Crear la red de Hopfield una vez, ya que los patrones no cambian
    hopfield_net = HopfieldNetwork(patterns_set)
    num_trials = 10  # Número de pruebas por configuración

    for noise_level in noise_levels:
        # Inicializar contadores de casos
        correct = incorrect = spurious = cycles = 0

        for _ in range(num_trials):
            # Seleccionar un patrón objetivo al azar
            target_pattern_idx = np.random.randint(len(patterns_set))
            # Añadir ruido al patrón objetivo
            noisy_pattern = add_noise(patterns_set[target_pattern_idx], noise_level=noise_level)
            # Recuperar el patrón
            recovered_pattern, states_history, energies = hopfield_net.get_similar(noisy_pattern.copy(), 100)
            # Categorizar el resultado
            result = categorize_result(recovered_pattern, patterns_set, states_history, target_pattern_idx)
            if result == 'Correcto':
                correct += 1
            elif result == 'Incorrecto':
                incorrect += 1
            elif result == 'Espurio':
                spurious += 1
            elif result == 'Ciclo':
                cycles += 1

        # Almacenar los resultados
        results_noise.append({
            'noise_level': noise_level,
            'correct': correct,
            'incorrect': incorrect,
            'spurious': spurious,
            'cycles': cycles
        })

    # Convertir resultados a DataFrame
    df_noise = pd.DataFrame(results_noise)
    df_noise['correct_percent'] = df_noise['correct'] / num_trials * 100
    df_noise['incorrect_percent'] = df_noise['incorrect'] / num_trials * 100
    df_noise['spurious_percent'] = df_noise['spurious'] / num_trials * 100
    df_noise['cycles_percent'] = df_noise['cycles'] / num_trials * 100

    # Graficar
    plt.figure(figsize=(10, 6))
    plt.plot(df_noise['noise_level'], df_noise['correct_percent'], label='Correcto')
    plt.plot(df_noise['noise_level'], df_noise['incorrect_percent'], label='Incorrecto')
    plt.plot(df_noise['noise_level'], df_noise['spurious_percent'], label='Espurio')
    plt.plot(df_noise['noise_level'], df_noise['cycles_percent'], label='Ciclo')
    plt.xlabel('Nivel de Ruido')
    plt.ylabel('Porcentaje de Casos')
    plt.title('Impacto del Nivel de Ruido (Ortogonalidad Constante entre 5 y 15)')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()