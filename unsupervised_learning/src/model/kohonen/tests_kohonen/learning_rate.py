from collections import defaultdict

import numpy as np
import pandas as pd

from TP4.src.model.kohonen.plotting_utils import visualize_clusters, plot_quantization_errors
from TP4.src.model.kohonen.som import SOM
from TP4.src.utils import standardize_inputs


def main():
    def constant_learning_rate(lr):
        return lambda x: lr

    def linear_decay_learning_rate(initial_lr, epochs):
        return lambda x: initial_lr * (1 - x / epochs)

    def exponential_decay_learning_rate(initial_lr, decay_rate):
        return lambda x: initial_lr * np.exp(-decay_rate * x)

    # Cargar los datos desde el archivo CSV
    data = pd.read_csv('../../../../inputs/europe.csv')  # Ajusta el nombre del archivo si es necesario
    np.set_printoptions(suppress=True, precision=5)  # Desactiva notación científica y ajusta los decimales a 5

    inputs = data[["Area", "GDP", "Inflation", "Life.expect", "Military", "Pop.growth", "Unemployment"]].to_numpy()

    standard_inputs = standardize_inputs(inputs)

    # Obtener los nombres de los países
    countries = data["Country"].to_numpy()

    # Inicializar y entrenar el SOM
    k = 5  # Ajusta esto si es necesario
    epochs = 500 * 1

    # Define different learning rate functions
    learning_rates = {
        'Constant LR (0.1)': constant_learning_rate(0.1),
        'Constant LR (0.5)': constant_learning_rate(0.5),
        'Linear Decay LR': linear_decay_learning_rate(0.5, epochs),
        'Exponential Decay LR': exponential_decay_learning_rate(0.5, 0.01)
    }

    # Initialize variables to store results
    quantization_errors_dict = {}

    for lr_name, lr_func in learning_rates.items():
        # Initialize and train the SOM
        som = SOM(k=5, learning_rate=lr_func)
        epochs = 100  # Adjust as needed
        grid, quantization_errors = som.train(standard_inputs, epochs)
        quantization_errors_dict[lr_name] = quantization_errors

    # Plot the quantization error for different learning rates
    plot_quantization_errors(quantization_errors_dict)


if __name__ == "__main__":
    main()
