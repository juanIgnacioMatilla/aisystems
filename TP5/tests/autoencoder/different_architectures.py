import random
import numpy as np
import matplotlib.pyplot as plt
import json

from TP5.src.model.autoencoder import Autoencoder
from TP5.src.model.autoencoder_adam import AdamAutoencoder
from TP5.src.model.autoencoder_vanilla import VanillaAutoencoder
from TP5.tests.utils import (
    to_bin_array, Font3, plot_max_error_char, interpolate_latent_vectors,
    plot_latent_space_with_interpolation, decode_and_plot_interpolated_chars
)


def run_training(X, layer_sizes, learning_rate, num_epochs, error_bar_interval=500):
    """
    Instancia y entrena el autoencoder.

    Args:
        X (np.ndarray): Datos de entrada con forma (num_samples, input_size).
        layer_sizes (list): Lista que especifica el tamaño de cada capa.
        learning_rate (float): Tasa de aprendizaje para el entrenamiento.
        num_epochs (int): Número de épocas de entrenamiento.
        error_bar_interval (int): Intervalo (en épocas) para registrar el error máximo de píxel.

    Returns:
        Autoencoder: Instancia del autoencoder entrenado.
        list: Lista del promedio de pérdidas por época.
        list: Lista de errores máximos de píxel en intervalos especificados.
    """
    autoencoder = Autoencoder(layer_sizes)
    #  = VanillaAutoencoder(layer_sizes)
    # autoencoder = AdamAutoencoder(layer_sizes)
    loss_history, max_pixel_error_history = autoencoder.train(
        X, learning_rate=learning_rate, num_epochs=num_epochs, error_bar_interval=error_bar_interval
    )
    return autoencoder, loss_history, max_pixel_error_history


if __name__ == "__main__":
    # Cargar parámetros desde config.json
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Parámetros generales
    NUM_RUNS = config['num_runs']
    LEARNING_RATE = config['learning_rate']
    NUM_EPOCHS = config['num_epochs']
    ERROR_BAR_EPOCH_INTERVAL = config['error_bar_epoch_interval']  # Intervalo para registrar el error máximo de píxel
    input_size = config['input_size']  # Por ejemplo, 35 para 7 filas * 5 columnas

    # Preparar los datos de entrada
    X = []
    for char in Font3:
        bin_array = to_bin_array(char)
        bin_vector = bin_array.flatten()
        X.append(bin_vector)
    X = np.array(X).astype(float)

    # Definir las configuraciones de capas
    layer_size_configs = [
        {'name': '15-5', 'hidden_layers': [15, 5], 'latent_size': 2},
        {'name': '25-15-5', 'hidden_layers': [25, 15, 5], 'latent_size': 2},
        {'name': '50-30', 'hidden_layers': [50, 30], 'latent_size': 2},
        {'name': '60-50-30', 'hidden_layers': [60, 50, 30], 'latent_size': 2},
    ]

    # Inicializar listas para almacenar los mínimos de pérdida y error de píxel y sus desviaciones estándar
    min_losses = []
    min_pixel_errors = []
    min_losses_std = []
    min_pixel_errors_std = []
    config_names = []

    for config in layer_size_configs:
        # Obtener el nombre y las capas ocultas de la configuración
        config_name = config['name']
        hidden_layers = config['hidden_layers']
        latent_size = config['latent_size']

        # Construir la arquitectura del autoencoder
        layer_sizes = [input_size] + hidden_layers + [latent_size] + hidden_layers[::-1] + [input_size]

        # Inicializar listas para almacenar métricas y modelos de todas las ejecuciones
        all_loss_histories = []
        all_max_pixel_error_histories = []
        all_autoencoders = []

        for run in range(NUM_RUNS):
            # Opcional: Establecer una semilla única para cada ejecución para reproducibilidad
            seed = config.get('seed', 42) + run  # Semilla por defecto es 42 si no se especifica
            np.random.seed(seed)
            random.seed(seed)
            print(f"Iniciando ejecución {run + 1}/{NUM_RUNS} para configuración '{config_name}' con semilla {seed}...")
            # Ejecutar entrenamiento
            autoencoder, loss_history, max_pixel_error_history = run_training(
                X, layer_sizes, LEARNING_RATE, NUM_EPOCHS, ERROR_BAR_EPOCH_INTERVAL
            )
            all_autoencoders.append(autoencoder)
            all_loss_histories.append(loss_history)
            all_max_pixel_error_histories.append(max_pixel_error_history)
            print(f"Ejecución {run + 1} para configuración '{config_name}' completada.")

        # Convertir listas a arreglos numpy para facilitar el cálculo
        all_loss_histories = np.array(all_loss_histories)  # Forma: (NUM_RUNS, NUM_EPOCHS)
        all_max_pixel_error_histories = np.array(all_max_pixel_error_histories)  # Forma: (NUM_RUNS, num_intervals)

        # Obtener las pérdidas y errores finales
        final_losses = all_loss_histories[:, -1]  # Pérdida al final de las épocas
        final_pixel_errors = all_max_pixel_error_histories[:, -1]  # Error de píxel al final

        # Calcular la pérdida y error de píxel medios finales
        mean_final_loss = np.mean(final_losses)
        mean_final_pixel_error = np.mean(final_pixel_errors)

        # Calcular la desviación estándar
        std_final_loss = np.std(final_losses)
        std_final_pixel_error = np.std(final_pixel_errors)

        # Almacenar los resultados
        min_losses.append(mean_final_loss)
        min_pixel_errors.append(mean_final_pixel_error)
        min_losses_std.append(std_final_loss)
        min_pixel_errors_std.append(std_final_pixel_error)
        config_names.append(config_name)

    # Graficar las pérdidas mínimas con barras de error
    plt.figure(figsize=(10, 6))
    plt.bar(config_names, min_losses, yerr=min_losses_std, capsize=5, color='skyblue')
    plt.xlabel('Configuración del Autoencoder')
    plt.ylabel('Valor Mínimo de Pérdida')
    plt.title('Comparación de Valores Mínimos de Pérdida para Diferentes Autoencoders')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Graficar los errores de píxel mínimos con barras de error
    plt.figure(figsize=(10, 6))
    plt.bar(config_names, min_pixel_errors, yerr=min_pixel_errors_std, capsize=5, color='salmon')
    plt.xlabel('Configuración del Autoencoder')
    plt.ylabel('Error Mínimo de Píxel')
    plt.title('Comparación de Errores Mínimos de Píxel para Diferentes Autoencoders')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()