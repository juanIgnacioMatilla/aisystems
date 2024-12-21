import numpy as np

from neural_networks.src.model.multilayer_perceptron.vanilla.multi_layer_perceptron import MultiLayerPerceptron
if __name__ == "__main__":
    # Multicapa
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Datos de entrenamiento
    y = np.array([[0], [1], [1], [0]])  # XOR como salida deseada
    for i in range(15):
        print("Run ",i)
        mlp = MultiLayerPerceptron([2, 4, 1], learning_rate=0.1)  # 2 neuronas de entrada, 2 ocultas, 1 de salida
        errors = mlp.train(X, y, epochs=10000)
        # Predicciones
        for x in X:
            print(f"Entrada: {x}, Predicción: {mlp.predict(x)}")
