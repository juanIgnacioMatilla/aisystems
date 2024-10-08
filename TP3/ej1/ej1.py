import json
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from TP3.src.model.simple_perceptron.simple_lineal_perceptron_classifier import SimplePerceptronClassifier


def set_up(file_path="./config.json"):
    with open(file_path, "r") as file:
        config = json.load(file)
    return config["epochs"], config["learning_rate"]


def main():
    # Set up
    epochs, learning_rate = set_up()
    # Datos
    # X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # y = np.array([0, 0, 0, 1])  # AND outputs
    #
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Datos de entrenamiento
    y = np.array([0, 1, 1, 0])  # XOR como salida deseada
    # Parámetros del entrenamiento
    initial_weights = np.random.rand(X.shape[1] + 1)  # Pesos aleatorios, incluye bias
    # Configuración de archivos temporales para las imágenes
    filenames = []

    # Entrenamiento del perceptrón y guardar gráficos en cada época
    perceptron_classifier = SimplePerceptronClassifier(learning_rate=learning_rate)
    for epoch in range(epochs):
        trained_neuron, errors_by_epoch = perceptron_classifier.train(X, y, initial_weights, epochs=1)
        # initial_weights = trained_neuron.weights
        # Gráfico de los puntos
        plt.figure()
        for i, label in enumerate(y):
            if label == 0:
                plt.scatter(X[i, 0], X[i, 1], color='red', label='y=0' if i == 0 else "")
            else:
                plt.scatter(X[i, 0], X[i, 1], color='green', label='y=1' if i == 1 else "")

        # Cálculo del hiperplano (solo si w2 != 0)
        w1, w2 = trained_neuron.weights[0], trained_neuron.weights[1]
        bias = trained_neuron.weights[-1]

        if w2 != 0:
            # Calcular los puntos de la línea que representan el hiperplano
            x1_values = np.array([0, 1])  # Valores extremos de x1 en el rango de datos
            x2_values = - (w1 * x1_values + bias) / w2
            plt.plot(x1_values, x2_values, color='blue', label='Hiperplano')

        # Fijar los límites de los ejes
        plt.xlim(0, 1.5)
        plt.ylim(0, 1.5)
        # Etiquetas y título
        plt.title(f'Epoch {epoch + 1}')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        plt.grid(True)

        # Guardar cada figura temporalmente
        filename = f'perceptron_epoch_{epoch + 1}.png'
        filenames.append(filename)
        plt.savefig(filename)
        plt.close()

    # Crear un video MP4
    video_filename = 'perceptron_xor_training.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Código para MP4
    video = cv2.VideoWriter(video_filename, fourcc, 1, (640, 480))  # 1 fps

    for filename in filenames:
        img = cv2.imread(filename)
        img = cv2.resize(img, (640, 480))  # Cambiar tamaño a 640x480
        video.write(img)

    video.release()  # Liberar el objeto de video

    # Eliminar los archivos temporales de imágenes
    for filename in filenames:
        os.remove(filename)

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])  # AND outputs
    initial_weights = np.random.rand(X.shape[1] + 1)  # Pesos aleatorios, incluye bias
    # Configuración de archivos temporales para las imágenes
    filenames = []

    # Entrenamiento del perceptrón y guardar gráficos en cada época
    perceptron_classifier = SimplePerceptronClassifier(learning_rate=learning_rate)
    for epoch in range(epochs):
        trained_neuron, errors_by_epoch = perceptron_classifier.train(X, y, initial_weights, epochs=1)
        # initial_weights = trained_neuron.weights
        # Gráfico de los puntos
        plt.figure()
        for i, label in enumerate(y):
            if label == 0:
                plt.scatter(X[i, 0], X[i, 1], color='red', label='y=0' if i == 0 else "")
            else:
                plt.scatter(X[i, 0], X[i, 1], color='green', label='y=1' if i == 1 else "")

        # Cálculo del hiperplano (solo si w2 != 0)
        w1, w2 = trained_neuron.weights[0], trained_neuron.weights[1]
        bias = trained_neuron.weights[-1]

        if w2 != 0:
            # Calcular los puntos de la línea que representan el hiperplano
            x1_values = np.array([0, 1])  # Valores extremos de x1 en el rango de datos
            x2_values = - (w1 * x1_values + bias) / w2
            plt.plot(x1_values, x2_values, color='blue', label='Hiperplano')

        # Fijar los límites de los ejes
        plt.xlim(0, 1.5)
        plt.ylim(0, 1.5)
        # Etiquetas y título
        plt.title(f'Epoch {epoch + 1}')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        plt.grid(True)

        # Guardar cada figura temporalmente
        filename = f'perceptron_epoch_{epoch + 1}.png'
        filenames.append(filename)
        plt.savefig(filename)
        plt.close()

    # Crear un video MP4
    video_filename = 'perceptron_and_training.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Código para MP4
    video = cv2.VideoWriter(video_filename, fourcc, 1, (640, 480))  # 1 fps

    for filename in filenames:
        img = cv2.imread(filename)
        img = cv2.resize(img, (640, 480))  # Cambiar tamaño a 640x480
        video.write(img)

    video.release()  # Liberar el objeto de video

    # Eliminar los archivos temporales de imágenes
    for filename in filenames:
        os.remove(filename)


if __name__ == "__main__":
    main()
