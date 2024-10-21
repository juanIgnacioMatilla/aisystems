import numpy as np
import pandas as pd
from collections import defaultdict

from TP4.src.model.kohonen.plotting_utils import create_interactive_plot, create_udm_plot
from TP4.src.model.kohonen.som import SOM
from TP4.src.utils import standardize_inputs


def main():
    # Cargar los datos desde el archivo CSV
    data = pd.read_csv('./inputs/europe.csv')  # Ajusta el nombre del archivo si es necesario
    np.set_printoptions(suppress=True, precision=5)  # Desactiva notación científica y ajusta los decimales a 5

    # Seleccionar las columnas que quieres usar para entrenar el SOM
    inputs = data[["Area", "GDP", "Inflation", "Life.expect", "Military", "Pop.growth", "Unemployment"]].to_numpy()
    social_inputs = data[["Life.expect", "Pop.growth", "Unemployment"]].to_numpy()
    economy_inputs = data[["GDP", "Inflation", "Unemployment"]].to_numpy()
    test_inputs = data[["Area"]].to_numpy()

    # Puedes cambiar la entrada aquí según lo que desees usar
    # inputs = test_inputs

    standard_inputs = standardize_inputs(inputs)

    # Obtener los nombres de los países
    countries = data["Country"].to_numpy()

    # Inicializar y entrenar el SOM
    k = 5  # Ajusta esto si es necesario
    som = SOM(k=k)
    epochs = 500 * 1
    grid, _ = som.train(standard_inputs, epochs)

    # Asignar países a neuronas
    neuron_countries = defaultdict(list)
    neuron_counts = np.zeros((k, k))

    for i, standardized_input in enumerate(standard_inputs):
        bmu = grid.find_bmu(standardized_input)
        print(countries[i])
        print(f'input         : {inputs[i]}')
        print(f'standard_input: {standardized_input}')
        print(f'grid.matrix_bm: {grid.matrix[bmu].weights}')
        print()
        neuron_counts[bmu] += 1

        # Asignar países a las neuronas
        if bmu not in neuron_countries:
            neuron_countries[bmu] = [countries[i]]
        else:
            neuron_countries[bmu].append(countries[i])

    # Paso 1: Encontrar las dimensiones máximas de la matriz
    max_fila = max(key[0] for key in neuron_countries)
    max_columna = max(key[1] for key in neuron_countries)

    # Paso 2: Crear una matriz vacía (lista de listas) con listas vacías en cada celda
    matriz = [[[] for _ in range(max_columna + 1)] for _ in range(max_fila + 1)]

    # Paso 3: Llenar la matriz con los valores del mapa
    for (fila, columna), paises in neuron_countries.items():
        matriz[fila][columna] = '\n '.join(paises)  # Concatenar los países con un salto de línea como separador

    print(matriz)
    print(neuron_counts)
    print("Neurona 0, 0: ", grid.matrix[0, 0].weights)
    print("Neurona 0, 1: ", grid.matrix[0, 1].weights)
    print("Neurona 1, 0: ", grid.matrix[1, 0].weights)
    print("Neurona 1, 1: ", grid.matrix[1, 1].weights)

    # Crear el gráfico interactivo
    create_interactive_plot(neuron_counts, matriz, "Life Expectancy")

    # Calcular y mostrar las distancias promedio
    average_distances = grid.calculate_average_distances()

    # Calcular y crear el UDM
    # Pasar k directamente como argumento
    create_udm_plot(average_distances, "UDM")

    print("Average Euclidean Distances to Neighbors:")
    print(average_distances)


if __name__ == "__main__":
    main()
