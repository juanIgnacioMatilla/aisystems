import numpy as np
import pandas as pd
import plotly.graph_objects as go
from collections import defaultdict
from TP4.src.model.kohonen.som import SOM


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
    inputs = test_inputs

    # Estandarizar entradas utilizando Z-score
    mean = np.mean(inputs, axis=0)
    std_dev = np.std(inputs, axis=0)
    standardized_inputs = (inputs - mean) / std_dev

    # Obtener los nombres de los países
    countries = data["Country"].to_numpy()

    # Inicializar y entrenar el SOM
    k = 5  # Ajusta esto si es necesario
    som = SOM(k=k)
    epochs = 500 * 1
    grid = som.train(standardized_inputs, epochs)

    # Asignar países a neuronas
    neuron_countries = defaultdict(list)
    neuron_counts = np.zeros((k, k))

    for i, standardized_input in enumerate(standardized_inputs):
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
    print("Neurona 0, 0: ",  grid.matrix[0, 0].weights)
    print("Neurona 0, 1: ",  grid.matrix[0, 1].weights)
    print("Neurona 1, 0: ",  grid.matrix[1, 0].weights)
    print("Neurona 1, 1: ",  grid.matrix[1, 1].weights)

    # Crear el gráfico interactivo
    create_interactive_plot(neuron_counts, matriz, "Life Expectancy")

    # Calcular y mostrar las distancias promedio
    average_distances = grid.calculate_average_distances()

    # Calcular y crear el UDM
    create_udm_plot(average_distances)  # Pasar k directamente como argumento

    print("Average Euclidean Distances to Neighbors:")
    print(average_distances)


def create_interactive_plot(neuron_counts, matriz, legend):
    data = neuron_counts

    # Crear el heatmap
    heatmap = go.Heatmap(
        z=data,
        text=matriz,  # Añadir la matriz de texto
        hovertemplate='count: %{z}<br>'
                      'x: %{x}<br>'
                      'y: %{y}<br>'
                      '%{text}<extra></extra>',  # Mostrar texto personalizado en el hover
        showscale=True
    )

    # Crear la figura
    fig = go.Figure(data=heatmap)
    # Añadir un título al gráfico
    fig.update_layout(title=legend)
    # Agregar bordes utilizando líneas
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            fig.add_shape(type='rect',
                          x0=j - 0.5, x1=j + 0.5,
                          y0=i - 0.5, y1=i + 0.5,
                          line=dict(color='black', width=1))

    # Configurar el layout para mostrar el heatmap correctamente
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
    )

    # Mostrar la figura
    fig.show()


def create_udm_plot(avg_distances):

    # Crear el heatmap de UDM
    udm_heatmap = go.Heatmap(
        z=avg_distances,
        colorscale='Greys',  # Escala de colores en blanco y negro
        colorbar=dict(title='Distance'),
        showscale=True
    )

    # Crear la figura
    fig = go.Figure(data=udm_heatmap)
    fig.update_layout(title="Unified Distance Matrix (UDM)")

    # Mostrar la figura
    fig.show()



if __name__ == "__main__":
    main()
