import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from TP4.hopfield_tests.main_hopfield import add_noise
from TP4.src.model.hopfield.hopfield_network import HopfieldNetwork
from TP4.src.model.hopfield.orthogonality import find_orthogonal_subset


def main():
    # Nivel de ruido constante
    noise_level = 0.2  # 20% de ruido

    # Valores de Min y Max Ortogonalidad a evaluar
    min_orthogonality_values = list(range(16))

    max_orthogonality = 13  # Mantener Max Ortogonalidad constante

    results_orthogonality = []
    num_trials = 10  # Número de pruebas por configuración

    for min_orthogonality in min_orthogonality_values:
        # Encontrar un conjunto de patrones con la ortogonalidad especificada
        letters, patterns_set = find_orthogonal_subset(min_orthogonality, max_orthogonality)
        if patterns_set is None:
            print(f"No se encontraron patrones para Min Ortogonalidad = {min_orthogonality}")
            continue

        # Crear la red de Hopfield con los patrones seleccionados
        hopfield_net = HopfieldNetwork(patterns_set)

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
        results_orthogonality.append({
            'min_orthogonality': min_orthogonality,
            'correct': correct,
            'incorrect': incorrect,
            'spurious': spurious,
            'cycles': cycles
        })

    # Convertir resultados a DataFrame
    df_orthogonality = pd.DataFrame(results_orthogonality)
    df_orthogonality['correct_percent'] = df_orthogonality['correct'] / num_trials * 100
    df_orthogonality['incorrect_percent'] = df_orthogonality['incorrect'] / num_trials * 100
    df_orthogonality['spurious_percent'] = df_orthogonality['spurious'] / num_trials * 100
    df_orthogonality['cycles_percent'] = df_orthogonality['cycles'] / num_trials * 100

    # Graficar
    plt.figure(figsize=(10, 6))
    plt.plot(df_orthogonality['min_orthogonality'], df_orthogonality['correct_percent'], label='Correcto')
    plt.plot(df_orthogonality['min_orthogonality'], df_orthogonality['incorrect_percent'], label='Incorrecto')
    plt.plot(df_orthogonality['min_orthogonality'], df_orthogonality['spurious_percent'], label='Espurio')
    plt.plot(df_orthogonality['min_orthogonality'], df_orthogonality['cycles_percent'], label='Ciclo')
    plt.xlabel('Min Ortogonalidad')
    plt.ylabel('Porcentaje de Casos')
    plt.title('Impacto de la Ortogonalidad (Nivel de Ruido Constante al 20%)')
    plt.legend()
    plt.show()

def categorize_result(recovered_pattern, patterns, states_history, target_pattern_idx):
    # Verificar si coincide con alguno de los patrones almacenados
    for idx, stored_pattern in enumerate(patterns):
        if np.array_equal(recovered_pattern, stored_pattern):
            if idx == target_pattern_idx:
                # print(f"CORRECTO")
                return 'Correcto'
            else:
                # print(f"INCORRECTO")
                return 'Incorrecto'
    # Verificar si hay un ciclo
    if len(states_history) > len(set(tuple(s) for s in states_history)):
        # print(f"CICLO")
        return 'Ciclo'
    # print(f"ESPURIO")
    return 'Espurio'

if __name__ == "__main__":
    main()