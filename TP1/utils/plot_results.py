import numpy as np
from matplotlib import pyplot as plt


def plot_scatter(results):
    methods = [result["method"] for result in results]
    times = np.array([result["time"] for result in results])
    expanded_nodes = np.array([result["expanded_nodes"] for result in results])
    errors = np.array([result["time_error"] for result in results])  # Supuesto campo para errores

    fig, ax = plt.subplots(figsize=(12, 8))  # Aumenta el tamaño del gráfico

    # Generar colores únicos para cada método y heurística
    unique_methods = list(set(methods))
    colors = plt.cm.get_cmap('tab10', len(unique_methods))

    # Diccionario para asociar cada método con un color
    color_dict = {method: colors(i) for i, method in enumerate(unique_methods)}

    # Graficar los puntos con barras de error
    for method in unique_methods:
        indices = [i for i, m in enumerate(methods) if m == method]
        ax.errorbar(
            times[indices],
            expanded_nodes[indices],
            xerr=errors[indices],  # Error en el tiempo
            fmt='o',
            color=color_dict[method],
            label=method,
            capsize=5,  # Tamaño de las "caps" en los extremos de las barras de error
            capthick=2  # Grosor de las "caps"
        )

    ax.set_xlabel('Time (seconds)', fontsize=14)  # Aumentar tamaño de fuente
    ax.set_ylabel('Expanded Nodes', fontsize=14)  # Aumentar tamaño de fuente
    ax.set_title('Time vs Expanded Nodes',
                 fontsize=16)  # Aumentar tamaño de fuente

    # Configurar la escala logarítmica para el eje x
    ax.set_xscale('log')

    # Configurar la leyenda en la parte superior derecha del gráfico
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fancybox=True, shadow=True,
              fontsize=14)  # Aumentar tamaño de fuente

    plt.tight_layout()  # Ajusta el diseño para que la leyenda se muestre correctamente
    plt.show()
