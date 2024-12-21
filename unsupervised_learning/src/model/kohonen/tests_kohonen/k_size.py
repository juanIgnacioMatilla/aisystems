from collections import defaultdict

import numpy as np
import pandas as pd

from TP4.src.model.kohonen.plotting_utils import plot_quantization_errors, create_interactive_plot
from TP4.src.model.kohonen.som import SOM
from TP4.src.utils import standardize_inputs


def main():
    # Cargar los datos desde el archivo CSV
    data = pd.read_csv('../../../../inputs/europe.csv')  # Ajusta el nombre del archivo si es necesario
    np.set_printoptions(suppress=True, precision=5)  # Desactiva notación científica y ajusta los decimales a 5

    inputs = data[["Area", "GDP", "Inflation", "Life.expect", "Military", "Pop.growth", "Unemployment"]].to_numpy()

    standard_inputs = standardize_inputs(inputs)

    # Obtener los nombres de los países
    countries = data["Country"].to_numpy()

    # Define different grid sizes to test
    grid_sizes = [3, 5, 7, 10]

    # Initialize variables to store results
    quantization_errors_dict = {}
    neuron_counts_dict = {}

    matrices = {}

    for size in grid_sizes:
        som = SOM(k=size)
        epochs = 100
        grid, quantization_errors = som.train(standard_inputs, epochs)
        quantization_errors_dict[f'Grid Size {size}x{size}'] = quantization_errors

        # Asignar países a neuronas
        neuron_countries = defaultdict(list)

        # Assign countries to neurons
        neuron_counts = np.zeros((size, size))
        for i, standardized_input in enumerate(standard_inputs):
            bmu = grid.find_bmu(standardized_input)
            neuron_counts[bmu] += 1

            # Asignar países a las neuronas
            if bmu not in neuron_countries:
                neuron_countries[bmu] = [countries[i]]
            else:
                neuron_countries[bmu].append(countries[i])

        neuron_counts_dict[size] = neuron_counts

        # Paso 1: Encontrar las dimensiones máximas de la matriz
        max_fila = max(key[0] for key in neuron_countries)
        max_columna = max(key[1] for key in neuron_countries)

        # Paso 2: Crear una matriz vacía (lista de listas) con listas vacías en cada celda
        matriz = [[[] for _ in range(max_columna + 1)] for _ in range(max_fila + 1)]

        # Paso 3: Llenar la matriz con los valores del mapa
        for (fila, columna), paises in neuron_countries.items():
            matriz[fila][columna] = '\n '.join(paises)  # Concatenar los países con un salto de línea como separador
            matrices[size] = matriz

    # Plot quantization errors for different grid sizes
    plot_quantization_errors(quantization_errors_dict)

    # Visualize the neuron counts for each grid size
    for size, counts in neuron_counts_dict.items():
        create_interactive_plot(counts, matrices[size], legend=f'Neuron Activation Count (Grid Size {size}x{size})')


if __name__ == "__main__":
    main()
