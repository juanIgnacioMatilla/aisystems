from collections import defaultdict

import numpy as np
import pandas as pd

from TP4.src.model.kohonen.plotting_utils import visualize_clusters
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

    som = SOM(k=5)
    epochs = 100
    grid, _ = som.train(standard_inputs, epochs)

    # Map inputs to BMUs
    neuron_countries = defaultdict(list)
    for i, standardized_input in enumerate(standard_inputs):
        bmu = grid.find_bmu(standardized_input)
        neuron_countries[bmu].append(countries[i])

    # Visualize clusters
    visualize_clusters(neuron_countries, grid_size=som.k)

if __name__ == "__main__":
    main()
