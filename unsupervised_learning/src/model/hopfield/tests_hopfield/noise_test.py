import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from TP4.hopfield_tests.main_hopfield import add_noise
from TP4.src.model.hopfield.hopfield_network import HopfieldNetwork
from TP4.src.model.hopfield.orthogonality import find_orthogonal_subset
from TP4.src.model.hopfield.tests_hopfield.orthogonality_test import categorize_result

def calculate_orthogonality(patterns):
    """Calculate the average orthogonality of a set of patterns."""
    num_patterns = len(patterns)
    orthogonality_sum = 0
    for i in range(num_patterns):
        for j in range(i + 1, num_patterns):
            orthogonality_sum += np.dot(patterns[i], patterns[j])
    num_comparisons = num_patterns * (num_patterns - 1) / 2
    return orthogonality_sum / num_comparisons if num_comparisons > 0 else 0

def run_experiment(k, noise_range, min_orthogonality, max_orthogonality, num_noise_levels):
    noise_levels = np.linspace(0, noise_range, num_noise_levels)
    patterns_set, selected_letters = find_orthogonal_subset(min_orthogonality, max_orthogonality)
    if patterns_set is None:
        print("No se encontraron patrones para los valores de ortogonalidad especificados.")
        exit()

    avg_orthogonality = calculate_orthogonality(patterns_set)
    print(f"Promedio de ortogonalidad de los patrones utilizados: {avg_orthogonality:.2f}")
    print(f"Letras utilizadas: {', '.join(selected_letters)}")

    all_results = []

    for _ in range(k):
        results_noise = []

        # Crear la red de Hopfield una vez, ya que los patrones no cambian
        hopfield_net = HopfieldNetwork(patterns_set)
        num_trials = 10  # Número de pruebas por configuración

        for noise_level in noise_levels:
            correcto = incorrecto = espurio = cycles = 0

            for _ in range(num_trials):
                target_pattern_idx = np.random.randint(len(patterns_set))
                noisy_pattern = add_noise(patterns_set[target_pattern_idx], noise_level=noise_level)
                recovered_pattern, states_history, energies = hopfield_net.get_similar(noisy_pattern.copy(), 100)
                result = categorize_result(recovered_pattern, patterns_set, states_history, target_pattern_idx)
                if result == 'Correcto':
                    correcto += 1
                elif result == 'Incorrecto':
                    incorrecto += 1
                elif result == 'Espurio':
                    espurio += 1
                elif result == 'Ciclo':
                    cycles += 1

            # Calcular porcentajes
            results_noise.append({
                'noise_level': noise_level,
                'correcto': correcto / num_trials * 100,
                'incorrecto': incorrecto / num_trials * 100,
                'espurio': espurio / num_trials * 100,
                'cycles': cycles / num_trials * 100
            })

        all_results.append(results_noise)

    # Convertir resultados a DataFrame
    df_list = [pd.DataFrame(results) for results in all_results]
    df_noise = pd.concat(df_list).groupby('noise_level').agg(['mean', 'std']).reset_index()

    return df_noise

def plot_results(df_noise, state, k, min_orthogonality, max_orthogonality, noise_range):
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust()
    plt.errorbar(df_noise['noise_level'] * 100, df_noise[(state, 'mean')],
                 yerr=df_noise[(state, 'std')], fmt='o', color='blue', alpha=0.5,
                 ecolor='blue', elinewidth=2, capsize=5, capthick=2)
    plt.plot(df_noise['noise_level'] * 100, df_noise[(state, 'mean')], color='blue', alpha=0.7)
    plt.xlabel('Nivel de Ruido (%)')
    plt.ylabel(f'Porcentaje de Estados {state.capitalize()}')
    plt.title(f'Impacto del Nivel de Ruido en Estados {state.capitalize()}')
    plt.xticks(np.arange(0, noise_range + 10, 10))

    plt.show()

def main():
    k = 10  # Number of times to run the experiment
    min_orthogonality = 0
    max_orthogonality = 10

    df_noise_correcto = run_experiment(k, 0.6, min_orthogonality, max_orthogonality, 7)
    plot_results(df_noise_correcto, 'correcto', k, min_orthogonality, max_orthogonality, 60)

    df_noise_espurio = run_experiment(k, 1.0, min_orthogonality, max_orthogonality, 11)
    plot_results(df_noise_espurio, 'espurio', k, min_orthogonality, max_orthogonality, 100)

    df_noise_incorrecto = run_experiment(k, 1.0, min_orthogonality, max_orthogonality, 11)
    plot_results(df_noise_incorrecto, 'incorrecto', k, min_orthogonality, max_orthogonality, 100)

if __name__ == "__main__":
    main()