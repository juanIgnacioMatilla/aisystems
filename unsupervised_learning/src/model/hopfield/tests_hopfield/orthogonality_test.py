import numpy as np
import itertools
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations

from TP4.src.model.hopfield.hopfield_network import HopfieldNetwork
from TP4.src.model.hopfield.orthogonality import get_matrix_of_letters, patterns


# Functions to calculate orthogonality and recover subsets
def calculate_orthogonality_stats(subset_patterns):
    """Calcula la ortogonalidad promedio y su desviación estándar entre todas las parejas de patrones en el subconjunto."""
    pairs = itertools.combinations(range(len(subset_patterns)), 2)
    orthogonality_values = []
    for i, j in pairs:
        product = np.dot(subset_patterns[i], subset_patterns[j])
        orthogonality_values.append(abs(product))
    return np.mean(orthogonality_values), np.std(orthogonality_values)


def print_spurious_matrix(matrix):
    """Prints the matrix with 1 as * and -1 as blanks."""
    for row in matrix:
        print(''.join('*' if x == 1 else ' ' for x in row))


def test_recall_accuracy(network, patterns, perturbation_level=0.1):
    """Prueba la precisión de recuperación de la red de Hopfield."""
    correct_retrievals = 0
    incorrect_retrievals = 0
    spurious_retrievals = 0
    cycle_retrievals = 0

    for pattern in patterns:
        # Introducir perturbación
        noisy_pattern = pattern.copy()
        num_flips = int(len(pattern) * perturbation_level)
        flip_indices = np.random.choice(len(pattern), num_flips, replace=False)
        noisy_pattern[flip_indices] *= -1  # Invertir bits

        # Recuperar patrón
        recovered_pattern, states_history, _ = network.get_similar(noisy_pattern)

        # Get the index of the original pattern
        target_pattern_idx = np.where(np.all(patterns == pattern, axis=1))[0][0]

        result = categorize_result(recovered_pattern, patterns, states_history, target_pattern_idx)
        if result == 'Correcto':
            correct_retrievals += 1
        elif result == 'Incorrecto':
            incorrect_retrievals += 1
        elif result == 'Espurio':
            spurious_retrievals += 1
        elif result == 'Ciclo':
            cycle_retrievals += 1

    total_patterns = len(patterns)
    return (correct_retrievals / total_patterns) * 100, (incorrect_retrievals / total_patterns) * 100, (
            spurious_retrievals / total_patterns) * 100, (cycle_retrievals / total_patterns) * 100


def categorize_result(recovered_pattern, patterns, states_history, target_pattern_idx):
    # Verificar si coincide con alguno de los patrones almacenados
    for idx, stored_pattern in enumerate(patterns):
        if np.array_equal(recovered_pattern, stored_pattern):
            if idx == target_pattern_idx:
                return 'Correcto'
            else:
                return 'Incorrecto'
    # Verificar si hay un ciclo
    if len(states_history) > len(set(tuple(s) for s in states_history)):
        return 'Ciclo'
    return 'Espurio'


def get_matrix_of_letters(letters_subset):
    """Convierte una lista de letras en una matriz numpy."""
    letters_matrix = []
    for letter in letters_subset:
        letters_matrix.append(patterns[letter])
    return np.vstack(letters_matrix)


# Generate combinations and collect data
letters = list(patterns.keys())
all_combinations = list(combinations(letters, 4))

# Parameters
num_samples = 500  # Número de subconjuntos a muestrear
np.random.seed(42)  # Para reproducibilidad

# Ensure not to exceed the total number of combinations
num_samples = min(num_samples, len(all_combinations))
sampled_indices = np.random.choice(len(all_combinations), num_samples, replace=False)

# Lists to store results
avg_orthogonality_list = []
stdev_orthogonality_list = []
accuracy_list = []
incorrect_accuracy_list = []
spurious_accuracy_list = []
cycle_accuracy_list = []

for idx in sampled_indices:
    subset_letters = all_combinations[idx]
    subset_patterns = get_matrix_of_letters(subset_letters)  # Retorna un array numpy

    # Calculate average orthogonality and standard deviation
    avg_orthogonality, stdev_orthogonality = calculate_orthogonality_stats(subset_patterns)

    # Initialize Hopfield network
    hopfield_net = HopfieldNetwork(subset_patterns)

    # Evaluate retrieval accuracy
    accuracy, incorrect_accuracy, spurious_accuracy, cycle_accuracy = test_recall_accuracy(hopfield_net, subset_patterns,
                                                                           perturbation_level=0.1)

    avg_orthogonality_list.append(avg_orthogonality)
    stdev_orthogonality_list.append(stdev_orthogonality)
    accuracy_list.append(accuracy)
    incorrect_accuracy_list.append(incorrect_accuracy)
    spurious_accuracy_list.append(spurious_accuracy)
    cycle_accuracy_list.append(cycle_accuracy)

# Create a DataFrame
df = pd.DataFrame({
    'Ortogonalidad_Promedio': avg_orthogonality_list,
    'Ortogonalidad_Stdev': stdev_orthogonality_list,
    'Precisión': accuracy_list,
    'Precisión_Incorrecta': incorrect_accuracy_list,
    'Precisión_Espuria': spurious_accuracy_list,
    'Precisión_Ciclo': cycle_accuracy_list
})

# Visualization
# Group by bins and calculate statistics
num_bins = 10
df['Ortogonalidad_Bin'] = pd.cut(df['Ortogonalidad_Promedio'], bins=num_bins)

grouped = df.groupby('Ortogonalidad_Bin').agg({
    'Precisión': ['mean', 'std'],
    'Ortogonalidad_Promedio': 'mean',
    'Precisión_Incorrecta': ['mean', 'std'],
    'Precisión_Espuria': ['mean', 'std'],
    'Precisión_Ciclo': ['mean', 'std'],
}).reset_index()

grouped.columns = ['Ortogonalidad_Bin', 'Precisión_Media', 'Precisión_Stdev',
                   'Ortogonalidad_Media', 'Precisión_Incorrecta_Media', 'Precisión_Incorrecta_Stdev',
                   'Precisión_Espuria_Media', 'Precisión_Espuria_Stdev', 'Precisión_Ciclo_Media', 'Precisión_Ciclo_Stdev']

# Bar plot for correct states
plt.figure(figsize=(12, 8))
plt.bar(grouped['Ortogonalidad_Bin'].astype(str), grouped['Precisión_Media'],
        yerr=grouped['Precisión_Stdev'], capsize=5, color='skyblue', edgecolor='black')
plt.xticks(rotation=45)
plt.title('Porcentaje de Estados Correctos vs. Conjunto de Ortogonalidad con Desviación Estándar')
plt.xlabel('Conjunto de Ortogonalidad Promedio')
plt.ylabel('Porcentaje de estados correctos (%)')
plt.tight_layout()
plt.grid(axis='y')
plt.show()

# Bar plot for incorrect states
plt.figure(figsize=(12, 8))
plt.bar(grouped['Ortogonalidad_Bin'].astype(str), grouped['Precisión_Incorrecta_Media'],
        yerr=grouped['Precisión_Incorrecta_Stdev'], capsize=5, color='salmon', edgecolor='black')
plt.xticks(rotation=45)
plt.title('Porcentaje de Estados Incorrectos vs. Conjunto de Ortogonalidad con Desviación Estándar')
plt.xlabel('Conjunto de Ortogonalidad Promedio')
plt.ylabel('Porcentaje de estados incorrectos (%)')
plt.tight_layout()
plt.grid(axis='y')
plt.show()

# Bar plot for spurious states
plt.figure(figsize=(12, 8))
plt.bar(grouped['Ortogonalidad_Bin'].astype(str), grouped['Precisión_Espuria_Media'],
        yerr=grouped['Precisión_Espuria_Stdev'], capsize=5, color='lightgreen', edgecolor='black')
plt.xticks(rotation=45)
plt.title('Porcentaje de Estados Espurios vs. Conjunto de Ortogonalidad con Desviación Estándar')
plt.xlabel('Conjunto de Ortogonalidad Promedio')
plt.ylabel('Porcentaje de estados espurios (%)')
plt.tight_layout()
plt.grid(axis='y')
plt.show()

# Bar plot for cycle states
plt.figure(figsize=(12, 8))
plt.bar(grouped['Ortogonalidad_Bin'].astype(str), grouped['Precisión_Ciclo_Media'],
        yerr=grouped['Precisión_Ciclo_Stdev'], capsize=5, color='lightcoral', edgecolor='black')
plt.xticks(rotation=45)
plt.title('Porcentaje de Estados en Ciclo vs. Conjunto de Ortogonalidad con Desviación Estándar')
plt.xlabel('Conjunto de Ortogonalidad Promedio')
plt.ylabel('Porcentaje de estados ciclos (%)')
plt.tight_layout()
plt.grid(axis='y')
plt.show()

# Calculate the sums of the percentages for each bin
bin_sums = grouped[
    ['Ortogonalidad_Bin', 'Precisión_Media', 'Precisión_Incorrecta_Media', 'Precisión_Espuria_Media', 'Precisión_Ciclo_Media']].copy()
bin_sums['Total_Precisión'] = bin_sums['Precisión_Media'] + bin_sums['Precisión_Incorrecta_Media'] + bin_sums[
    'Precisión_Espuria_Media'] + bin_sums['Precisión_Ciclo_Media']

# Print the sums for each bin
print("Bin de Ortogonalidad | Suma de Porcentajes de Estados")
print("----------------------------------------------------")
for index, row in bin_sums.iterrows():
    print(f"{row['Ortogonalidad_Bin']}: {row['Total_Precisión']:.2f}% (Correctos: {row['Precisión_Media']:.2f}%, "
          f"Incorrectos: {row['Precisión_Incorrecta_Media']:.2f}%, Espurios: {row['Precisión_Espuria_Media']:.2f}%, Ciclos: {row['Precisión_Ciclo_Media']:.2f}%)")