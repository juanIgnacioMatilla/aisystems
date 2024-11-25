import numpy as np
from matplotlib import pyplot as plt
import json

from TP5.src.model.denoising_autoencoder import DenoisingAutoencoder
from TP5.tests.utils import add_noise, Font3, to_bin_array, plot_denoising_results

if __name__ == "__main__":
    # Cargar parámetros desde config.json
    with open('config.json', 'r') as f:
        config = json.load(f)

    X = []
    labels = []  # Lista para almacenar etiquetas de caracteres
    for char in Font3:
        bin_array = to_bin_array(char)
        bin_vector = bin_array.flatten()
        X.append(bin_vector)
        labels.append(char)  # Asumiendo que 'char' es la etiqueta del carácter
    X = np.array(X).astype(float)

    # Definir la arquitectura
    input_size = config['input_size']
    hidden_layers = config['hidden_layers']
    latent_size = config['latent_size']
    layer_sizes = [input_size] + hidden_layers + [latent_size] + hidden_layers[::-1] + [input_size]

    # Parámetros de entrenamiento
    learning_rate = config['learning_rate']
    num_epochs = config['num_epochs']
    num_runs = config['num_runs']  # Número de ejecuciones por nivel de ruido

    # Niveles de ruido a probar
    noise_levels = config['noise_levels']  # Ejemplo: [0.1, 0.3, 0.5]

    for noise_level in noise_levels:
        print(f"\nEntrenando con nivel de ruido: {noise_level * 100}%")
        loss_histories = []
        min_losses = []
        best_final_loss = np.inf
        best_autoencoder = None
        X_noisy_all_runs = None  # Para almacenar los datos ruidosos de la mejor ejecución

        for run in range(num_runs):
            print(f"Ejecución {run + 1}/{num_runs}")
            # Instanciar el autoencoder
            autoencoder = DenoisingAutoencoder(layer_sizes)

            # Generar datos ruidosos
            X_noisy = add_noise(X, noise_level)

            # Entrenar el autoencoder
            loss_history = autoencoder.train(X_noisy, X, learning_rate=learning_rate, num_epochs=num_epochs)
            loss_histories.append(loss_history)

            # Almacenar la pérdida mínima de esta ejecución
            min_loss = np.min(loss_history)
            min_losses.append(min_loss)

            # Verificar si esta es la mejor ejecución hasta ahora
            final_loss = loss_history[-1]
            if final_loss < best_final_loss:
                best_final_loss = final_loss
                best_autoencoder = autoencoder
                X_noisy_all_runs = X_noisy  # Almacenar los datos ruidosos de la mejor ejecución

        # Después de num_runs, calcular media y varianza de pérdidas mínimas
        mean_min_loss = np.mean(min_losses)
        var_min_loss = np.var(min_losses)
        print(f"Pérdida mínima promedio: {mean_min_loss:.6f}, Varianza: {var_min_loss:.6f}")

        # Convertir loss_histories a arreglo numpy para graficar
        loss_histories = np.array(loss_histories)  # Forma (num_runs, num_epochs)
        mean_loss_history = np.mean(loss_histories, axis=0)
        std_loss_history = np.std(loss_histories, axis=0)

        # Graficar mean_loss_history con barras de error
        epochs = np.arange(num_epochs)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, mean_loss_history, label='Pérdida Media')
        plt.fill_between(epochs, mean_loss_history - std_loss_history, mean_loss_history + std_loss_history, alpha=0.3, label='Desv. Est.')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida Promedio')
        plt.title(f'Pérdida de Entrenamiento (Nivel de Ruido: {noise_level * 100}%)')
        plt.legend()
        plt.show()

        # Reconstruir las imágenes utilizando el mejor modelo
        reconstructed_chars = []
        for i in range(X.shape[0]):
            x_noisy = X_noisy_all_runs[i].reshape(-1, 1)
            output = best_autoencoder.reconstruct(x_noisy)
            reconstructed = (output > 0.5).astype(int)
            reconstructed_chars.append(reconstructed.flatten())
        num_examples = config['num_examples']  # Número de ejemplos a mostrar
        plot_denoising_results(X[:num_examples], X_noisy_all_runs[:num_examples], reconstructed_chars[:num_examples],
                               num_examples)

        # Evaluar rendimiento
        total_incorrect = 0
        for i in range(X.shape[0]):
            total_incorrect += np.sum(np.abs(reconstructed_chars[i] - X[i]))
        avg_incorrect = total_incorrect / X.shape[0]
        print(f"Número promedio de píxeles incorrectos por imagen: {avg_incorrect}")

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
