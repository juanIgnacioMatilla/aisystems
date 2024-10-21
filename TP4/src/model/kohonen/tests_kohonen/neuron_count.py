from collections import defaultdict

import numpy as np
import pandas as pd

from TP4.src.model.kohonen.plotting_utils import plot_quantization_errors, create_interactive_plot, \
    create_udm_plot
from TP4.src.model.kohonen.som import SOM
from TP4.src.model.kohonen.topology import Topology
from TP4.src.utils import standardize_inputs


def main():
    # Cargar los datos desde el archivo CSV
    data = pd.read_csv('../../../../inputs/europe.csv')  # Ajusta el nombre del archivo si es necesario
    np.set_printoptions(suppress=True, precision=5)  # Desactiva notación científica y ajusta los decimales a 5

    inputs = data[["Area", "GDP", "Inflation", "Life.expect", "Military", "Pop.growth", "Unemployment"]].to_numpy()
    standard_inputs = standardize_inputs(inputs)

    # Obtener los nombres de los países
    countries = data["Country"].to_numpy()

    # Define topologies to test
    topologies = [Topology.RECTANGULAR, Topology.HEXAGONAL]

    for topo in topologies:
        som = SOM(k=5, topology=topo)
        epochs = 100
        grid, _ = som.train(standard_inputs, epochs)

        # Asignar países a neuronas
        neuron_countries = defaultdict(list)

        # Assign countries to neurons
        neuron_counts = np.zeros((5, 5))
        for i, standardized_input in enumerate(standard_inputs):
            bmu = grid.find_bmu(standardized_input)
            neuron_counts[bmu] += 1

            # Asignar países a las neuronas
            if bmu not in neuron_countries:
                neuron_countries[bmu] = [countries[i]]
            else:
                neuron_countries[bmu].append(countries[i])

        # Calculate U-Matrix
        average_distances = grid.calculate_average_distances()

        # Paso 1: Encontrar las dimensiones máximas de la matriz
        max_fila = max(key[0] for key in neuron_countries)
        max_columna = max(key[1] for key in neuron_countries)

        # Paso 2: Crear una matriz vacía (lista de listas) con listas vacías en cada celda
        matriz = [[[] for _ in range(max_columna + 1)] for _ in range(max_fila + 1)]

        # Paso 3: Llenar la matriz con los valores del mapa
        for (fila, columna), paises in neuron_countries.items():
            matriz[fila][columna] = '\n '.join(paises)  # Concatenar los países con un salto de línea como separador

        # Visualize neuron counts
        create_interactive_plot(neuron_counts, matriz,
                                legend=f'Neuron Activation Count ({topo.value.capitalize()} Topology)')

        # Visualize U-Matrix
        create_udm_plot(average_distances, title=f'UDM ({topo.value.capitalize()} Topology)')


if __name__ == "__main__":
    main()
