import numpy as np
from matplotlib import pyplot as plt


def standardize_inputs(inputs):
    # Estandarizar entradas utilizando Z-score
    mean = np.mean(inputs, axis=0)
    std_dev = np.std(inputs, axis=0)
    return (inputs - mean) / std_dev


def resize_figure(x, y, scale=1.0):
    """
    Ajusta el tamaño de la figura y los elementos internos proporcionalmente.

    Parámetros:
    - x, y: Figsize en pulgadas (ancho, alto).
    - scale: Factor de escala para mantener proporción de los elementos internos.
    """
    fig, ax = plt.subplots(figsize=(x, y))

    # Ajustamos el tamaño de la fuente y otros elementos proporcionales al tamaño de la figura
    ax.title.set_size(12 * scale)
    ax.xaxis.label.set_size(10 * scale)
    ax.yaxis.label.set_size(10 * scale)
    ax.tick_params(axis='both', which='major', labelsize=8 * scale)

    # Un ejemplo simple para mostrar un gráfico
    x_data = range(10)
    y_data = [i ** 2 for i in x_data]
    ax.plot(x_data, y_data, linewidth=2 * scale)

    plt.show()
