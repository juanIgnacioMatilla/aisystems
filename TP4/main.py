import numpy as np
import pandas as pd
from TP4.src.model.kohonen.som import SOM
from TP4.src.model.kohonen.topology import Topology


def main():
    # Cargar los datos desde el archivo CSV
    data = pd.read_csv('./inputs/europe.csv')  # Ajusta el nombre del archivo si es necesario
    np.set_printoptions(suppress=True, precision=2)  # Desactiva notación científica y ajusta los decimales a 2

    # Seleccionar las columnas que quieres usar para entrenar el SOM
    inputs = data[["Area", "GDP", "Inflation", "Life.expect", "Military", "Pop.growth", "Unemployment"]].to_numpy()  # Ejemplo con las columnas 'GDP' y 'Life.expect'

    # Initialize and train the SOM
    som = SOM(k=6)
    # vector of 7 numbers per input (input neurons)
    epochs = 500*7
    grid, inputs_per_neuron = som.train(inputs, epochs)
    print("Last epoch:")
    for neuron_coords, inputs in inputs_per_neuron[-1].items():
        print(f"Neuron coords: {neuron_coords}:")
        for input_vector in inputs:
            print(input_vector)
        print("---------")
if __name__ == "__main__":
     main()
