import numpy as np


def oja_pc1(X, learning_rate=0.1, n_iterations=1000):
    # Inicializar el peso aleatoriamente
    w = np.random.uniform(0, 1, size=X.shape[1])
    initial_learning_rate = learning_rate
    for epoch in range(n_iterations):
        learning_rate = initial_learning_rate / (epoch + 1)
        # Actualizar el vector de pesos usando la regla de Oja
        for x in X:
            # Calcular la proyección de x sobre w
            projection = np.dot(x, w)
            # Actualizar w
            w += learning_rate * projection * (x - projection * w)
    return w
