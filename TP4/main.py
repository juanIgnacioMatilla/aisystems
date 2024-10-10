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

    # decreasing_radius = lambda x: int(k - ((x / (epochs - 1)) * (k - 1)))
    # Initialize and train the SOM
    k = 6
    som = SOM(k=k, topology=Topology.RECTANGULAR, learning_rate=lambda x: 1/(x+1), radius=lambda x: 1)
    epochs = 1000
    grid, inputs_per_neuron = som.train(inputs, epochs)
    print("ULTIMA EPOCH")
    for neuron_coords, inputs in inputs_per_neuron[-1].items():
        print(f"Neurona {neuron_coords}:")
        for input_vector in inputs:
            print(input_vector)
        print("---------")
    print("PRIMERA EPOCH:")
    for neuron_coords, inputs in inputs_per_neuron[0].items():
        print(f"Neurona {neuron_coords}:")
        for input_vector in inputs:
            print(input_vector)
        print("---------")
if __name__ == "__main__":
     main()
