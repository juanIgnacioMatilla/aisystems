import numpy as np
from matplotlib import pyplot as plt
def plot_scatter(results):
    methods = [result["method"] for result in results]
    times = np.array([result["time"] for result in results])
    expanded_nodes = np.array([result["expanded_nodes"] for result in results])

    fig, ax = plt.subplots(figsize=(12, 8))  # Aumenta el tamaño del gráfico

    # Generar colores únicos para cada método y heurística
    unique_methods = list(set(methods))
    colors = plt.cm.get_cmap('tab10', len(unique_methods))

    # Diccionario para asociar cada método con un color
    color_dict = {method: colors(i) for i, method in enumerate(unique_methods)}

    # Graficar los puntos
    for method in unique_methods:
        indices = [i for i, m in enumerate(methods) if m == method]
        ax.scatter(times[indices], expanded_nodes[indices], color=color_dict[method], label=method)

    ax.set_xlabel('Time (seconds)', fontsize=14)  # Aumentar tamaño de fuente
    ax.set_ylabel('Expanded Nodes', fontsize=14)  # Aumentar tamaño de fuente
    ax.set_title('Time vs Expanded Nodes for Different Search Methods and Heuristics', fontsize=16)  # Aumentar tamaño de fuente

    # Configurar la leyenda a la derecha del gráfico
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True, fontsize=12)  # Aumentar tamaño de fuente

    plt.tight_layout()  # Ajusta el diseño para que la leyenda se muestre correctamente
    plt.show()

