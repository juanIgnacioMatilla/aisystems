import random
import numpy as np
import matplotlib.pyplot as plt
import json

from deep_learning.src.model.autoencoder import Autoencoder
from deep_learning.src.model.autoencoder_adam import AdamAutoencoder
from deep_learning.src.model.autoencoder_vanilla import VanillaAutoencoder
from deep_learning.tests.utils import (
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

    Returns:autoencoder
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

    # Parámetros para múltiples ejecuciones
    NUM_RUNS = config['num_runs']
    LEARNING_RATE = config['learning_rate']
    NUM_EPOCHS = config['num_epochs']
    ERROR_BAR_EPOCH_INTERVAL = config['error_bar_epoch_interval']  # Intervalo para registrar el error máximo de píxel

    # Preparar los datos de entrada
    X = []
    for char in Font3:
        bin_array = to_bin_array(char)
        bin_vector = bin_array.flatten()
        X.append(bin_vector)
    X = np.array(X).astype(float)

    # Definir la arquitectura
    input_size = config['input_size']  # Por ejemplo, 35 para 7 filas * 5 columnas
    hidden_layers = config['hidden_layers']  # Lista de tamaños de capas ocultas antes de la capa latente
    latent_size = config['latent_size']  # Dimensión del espacio latente
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
        print(f"Iniciando ejecución {run + 1}/{NUM_RUNS} con semilla {seed}...")
        # Ejecutar entrenamiento
        autoencoder, loss_history, max_pixel_error_history = run_training(
            X, layer_sizes, LEARNING_RATE, NUM_EPOCHS, ERROR_BAR_EPOCH_INTERVAL
        )
        all_autoencoders.append(autoencoder)
        all_loss_histories.append(loss_history)
        all_max_pixel_error_histories.append(max_pixel_error_history)
        print(f"Ejecución {run + 1} completada.")

    # Convertir listas a arreglos numpy para facilitar el cálculo
    all_loss_histories = np.array(all_loss_histories)  # Forma: (NUM_RUNS, NUM_EPOCHS)
    all_max_pixel_error_histories = np.array(all_max_pixel_error_histories)  # Forma: (NUM_RUNS, num_intervals)

    # Calcular media y desviación estándar para la pérdida
    mean_loss = np.mean(all_loss_histories, axis=0)
    std_loss = np.std(all_loss_histories, axis=0)

    # Calcular media y desviación estándar para el error máximo de píxel
    mean_max_pixel_error = np.mean(all_max_pixel_error_histories, axis=0)
    std_max_pixel_error = np.std(all_max_pixel_error_histories, axis=0)

    # Graficar el historial de pérdidas con barras de error (área sombreada)
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    epochs = np.arange(1, NUM_EPOCHS + 1)
    plt.plot(epochs, mean_loss, label='Pérdida Media', color='blue')
    plt.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, color='blue', alpha=0.2, label='Desv. Est.')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida Promedio')
    plt.title('Pérdida de Entrenamiento con Barras de Error')
    plt.legend()

    # Graficar el historial del error máximo de píxel con barras de error
    plt.subplot(1, 2, 2)
    interval_epochs = np.arange(ERROR_BAR_EPOCH_INTERVAL, NUM_EPOCHS + 1, ERROR_BAR_EPOCH_INTERVAL)
    plt.plot(interval_epochs, mean_max_pixel_error, marker='o', label='Error Máx. de Píxel Medio', color='red')
    plt.fill_between(interval_epochs,
                     mean_max_pixel_error - std_max_pixel_error,
                     mean_max_pixel_error + std_max_pixel_error,
                     color='red', alpha=0.2, label='Desv. Est.')
    plt.xlabel('Épocas')
    plt.ylabel('Error Máximo de Píxel')
    plt.title('Error Máximo de Píxel a través de las Épocas con Barras de Error')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Identificar la ejecución con el menor error máximo de píxel final
    final_max_pixel_errors = [history[-1] for history in all_max_pixel_error_histories]
    best_run_index = np.argmin(final_max_pixel_errors)
    best_autoencoder = all_autoencoders[best_run_index]
    print(f"La mejor ejecución es la {best_run_index + 1} con un error máximo de píxel final de: {final_max_pixel_errors[best_run_index]}")

    # Opcional: Guardar el mejor modelo
    # joblib.dump(best_autoencoder, 'best_autoencoder.pkl')
    # print("Modelo del mejor autoencoder guardado como 'best_autoencoder.pkl'.")

    # Reconstruir los caracteres utilizando el mejor autoencoder entrenado
    reconstructed_chars = []
    for i in range(X.shape[0]):
        x = X[i].reshape(-1, 1)
        output = best_autoencoder.reconstruct(x)
        # Umbralizar la salida
        reconstructed = (output > 0.5).astype(int)
        reconstructed_chars.append(reconstructed.flatten())

    # Calcular errores de píxel para todos los caracteres
    pixel_errors = []
    for i in range(X.shape[0]):
        x = X[i]
        reconstructed = reconstructed_chars[i]
        pixel_error = np.sum(np.abs(reconstructed - x))
        pixel_errors.append(pixel_error)

    # Identificar el carácter con el máximo error de píxel
    max_error_index = np.argmax(pixel_errors)
    print(f"El carácter con el error máximo de píxel está en el índice: {max_error_index}")

    # Graficar el carácter con el máximo error de píxel
    plot_max_error_char(max_error_index, reconstructed_chars, X, pixel_errors)

    # Extraer representaciones latentes de la mejor ejecución
    latent_representations = []
    for i in range(X.shape[0]):
        x = X[i].reshape(-1, 1)
        activations, _ = best_autoencoder.forward(x)
        latent = activations[len(hidden_layers) + 1]  # Índice de la capa latente en activations
        latent_representations.append(latent.flatten())
    latent_representations = np.array(latent_representations)

    # Crear etiquetas para los caracteres
    labels = [chr(i) for i in range(96, 96 + Font3.shape[0])]

    # Graficar el espacio latente
    plt.figure(figsize=(12, 10))
    plt.scatter(latent_representations[:, 0], latent_representations[:, 1], color='blue')

    # Anotar cada punto con su carácter correspondiente
    for i, label in enumerate(labels):
        plt.annotate(label, (latent_representations[i, 0], latent_representations[i, 1]),
                     fontsize=12, textcoords="offset points", xytext=(0, 5), ha='center')

    plt.xlabel('Dimensión Latente 1')
    plt.ylabel('Dimensión Latente 2')
    plt.title('Representación en el Espacio Latente de los Caracteres')
    plt.grid(True)
    plt.show()

    # Seleccionar 5 índices aleatorios para inspección visual
    random_indices = random.sample(range(X.shape[0]), 5)

    fig, axes = plt.subplots(5, 2, figsize=(6, 12))

    for idx, (ax_orig, ax_recon) in zip(random_indices, axes):
        # Carácter original
        original = X[idx].reshape(7, 5)
        # Carácter reconstruido
        x = X[idx].reshape(-1, 1)
        output = best_autoencoder.reconstruct(x)
        reconstructed = (output > 0.5).astype(int).reshape(7, 5)

        # Graficar original
        ax_orig.imshow(original, cmap='Greys')
        ax_orig.axis('off')
        ax_orig.set_title("Original")

        # Graficar reconstruido
        ax_recon.imshow(reconstructed, cmap='Greys')
        ax_recon.axis('off')
        ax_recon.set_title("Reconstruido")

    plt.tight_layout()
    plt.show()

    # Generar un vector latente aleatorio y decodificar
    latent_vector = np.random.uniform(-1, 1, (latent_size, 1))
    decoded_output = best_autoencoder.decode(latent_vector)
    reconstructed = (decoded_output > 0.5).astype(int).flatten()
    reconstructed_char = reconstructed.reshape(7, 5)

    # Interpolación entre dos vectores latentes
    if X.shape[0] >= 3:  # Asegurar que hay al menos 3 caracteres
        latent1 = latent_representations[1]
        latent2 = latent_representations[2]

        # Realizar interpolación
        interpolated_latents = interpolate_latent_vectors(latent1, latent2, steps=10)
        print(f"Generados {len(interpolated_latents)} vectores latentes interpolados.")

        # Graficar el espacio latente con interpolación
        plot_latent_space_with_interpolation(
            latent_representations, labels, 1, 2, interpolated_latents, latent1, latent2
        )

        # Graficar los caracteres interpolados
        decode_and_plot_interpolated_chars(best_autoencoder, interpolated_latents, title_prefix="Interp")
    else:
        print("No hay suficientes caracteres para realizar interpolación.")
