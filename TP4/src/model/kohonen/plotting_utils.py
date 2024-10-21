import numpy as np
from matplotlib import pyplot as plt
import plotly.graph_objects as go


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


def create_udm_plot(avg_distances, title):
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
    fig.update_layout(title=title)

    # Mostrar la figura
    fig.show()


def plot_quantization_errors(quantization_errors_dict):
    for lr_name, errors in quantization_errors_dict.items():
        plt.plot(errors, label=lr_name)

    plt.title('Quantization Error over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Average Quantization Error')
    plt.legend()
    plt.show()


def visualize_clusters(neuron_countries, grid_size):
    # Create a matrix to hold the country names
    cluster_matrix = [['' for _ in range(grid_size)] for _ in range(grid_size)]
    for (i, j), country_list in neuron_countries.items():
        cluster_matrix[i][j] = '\n'.join(country_list)

    # Create a heatmap with text annotations
    heatmap = go.Heatmap(
        z=np.zeros((grid_size, grid_size)),  # We can use zeros since we're focusing on text annotations
        text=cluster_matrix,
        texttemplate='%{text}',
        colorscale='Blues',
        showscale=False
    )

    fig = go.Figure(data=heatmap)
    fig.update_layout(title='Clusters on SOM Grid')
    fig.show()