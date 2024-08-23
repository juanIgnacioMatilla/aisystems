import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter


def plot_scatter(results, runs):
    methods = [result["method"] for result in results]
    times = np.array([result["time"] for result in results])
    expanded_nodes = np.array([result["expanded_nodes"] for result in results])
    errors = np.array([result["time_error"] for result in results])

    fig, ax = plt.subplots(figsize=(12, 8))

    unique_methods = list(set(methods))
    colors = plt.cm.get_cmap('tab10', len(unique_methods))
    color_dict = {method: colors(i) for i, method in enumerate(unique_methods)}

    for method in unique_methods:
        indices = [i for i, m in enumerate(methods) if m == method]
        ax.errorbar(
            times[indices],
            expanded_nodes[indices],
            xerr=errors[indices],
            fmt='o',
            color=color_dict[method],
            label=method,
            capsize=5,
            capthick=2
        )

    ax.set_xlabel('Time (seconds)', fontsize=14)
    ax.set_ylabel('Expanded Nodes', fontsize=14)
    ax.set_title(f'Time vs Expanded Nodes (Runs: {runs})', fontsize=16)

    # Cambiar a escala lineal para evitar la notación científica
    ax.set_xscale('linear')

    # Formatear el eje x para mostrar los valores en milisegundos sin notación científica
    # ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))

    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fancybox=True, shadow=True, fontsize=14)
    plt.tight_layout()
    plt.show()

