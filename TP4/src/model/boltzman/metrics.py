import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from TP4.src.model.boltzman.deep_belief_network import DBN
from TP4.src.model.boltzman.restricted_boltzmann_machine import RBM
from tensorflow.keras.datasets import mnist


def binarize_images(images, threshold=0.5):
    return (images > threshold).astype(np.float32)


def load_mnist_data_split():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Normalizar y binarizar las imágenes
    x_train = x_train.astype('float32') / 255
    x_train = binarize_images(x_train)
    x_train = x_train.reshape((x_train.shape[0], -1))

    x_test = x_test.astype('float32') / 255
    x_test = binarize_images(x_test)
    x_test = x_test.reshape((x_test.shape[0], -1))

    return x_train, y_train, x_test, y_test


def add_noise_to_image(image, noise_level=0.1):
    """
    Agrega ruido binomial a una imagen binarizada.

    :param image: Vector de la imagen original (valores 0 o 1).
    :param noise_level: Porcentaje de píxeles a los que se les agregará ruido.
    :return: Imagen con ruido.
    """
    noisy_image = image.copy()
    n_pixels = len(image)
    n_noisy = int(noise_level * n_pixels)
    noisy_indices = np.random.choice(n_pixels, n_noisy, replace=False)
    noisy_image[noisy_indices] = 1 - noisy_image[noisy_indices]
    return noisy_image


def mean_squared_error(original, reconstructed):
    """
    Calcula el Error Cuadrático Medio (MSE) entre las imágenes originales y reconstruidas.

    :param original: Array de forma (n_samples, n_features) de las imágenes originales.
    :param reconstructed: Array de forma (n_samples, n_features) de las imágenes reconstruidas.
    :return: MSE promedio.
    """
    return np.mean((original - reconstructed) ** 2)


if __name__ == "__main__":
    # Cargar los datos con separación de entrenamiento y prueba
    x_train, y_train, x_test, y_test = load_mnist_data_split()
    n_samples_train, n_visible = x_train.shape
    n_samples_test, _ = x_test.shape

    # Definir la estructura de la DBN
    layer_sizes = [784, 500, 200, 60]  # Puedes ajustar estos valores
    dbn = DBN(layer_sizes)
    rbm_individual = RBM(n_visible=784, n_hidden=60)
    epochs = 10

    # Inicializar listas para almacenar el MSE por época
    mse_dbn_list = []
    mse_rbm_list = []

    for epoch in range(1, epochs + 1):
        print(f"\n=== Época {epoch}/{epochs} ===")

        # Preentrenar la DBN una época
        print("Preentrenando la DBN...")
        dbn.pretrain(x_train, epochs=1, batch_size=100, learning_rate=0.01, k=1)

        # Reconstruir imágenes usando la DBN
        print("\nReconstruyendo imágenes con la DBN...")
        # Reconstrucción en batch para eficiencia
        reconstructions_dbn = dbn.reconstruct(x_test)
        mse_dbn = mean_squared_error(x_test, reconstructions_dbn)
        mse_dbn_list.append(mse_dbn)
        print(f"MSE de la DBN en la Época {epoch}: {mse_dbn}")

        # Entrenar la RBM individual una época
        print("\nEntrenando la RBM individual...")
        rbm_individual.train(x_train, epochs=1, batch_size=100, learning_rate=0.01, k=1)

        # Reconstruir imágenes usando la RBM individual
        print("\nReconstruyendo imágenes con la RBM individual...")
        # Reconstrucción en batch para eficiencia
        reconstructions_rbm = rbm_individual.reconstruct(x_test)
        mse_rbm = mean_squared_error(x_test, reconstructions_rbm)
        mse_rbm_list.append(mse_rbm)
        print(f"MSE de la RBM individual en la Época {epoch}: {mse_rbm}")

        # Comparar los MSE
        if mse_dbn < mse_rbm:
            print("\nLa DBN reconstruye las imágenes con menor error que la RBM individual.")
        else:
            print("\nLa RBM individual reconstruye las imágenes con menor error que la DBN.")

        # Evaluar Exactitud de Clasificación usando RBM
        print("\nEvaluando Exactitud de Clasificación usando RBM...")
        features_rbm_train = rbm_individual.transform(x_train)
        features_rbm_test = rbm_individual.transform(x_test)
        classifier_rbm = LogisticRegression(max_iter=1000)
        classifier_rbm.fit(features_rbm_train, y_train)
        predictions_rbm = classifier_rbm.predict(features_rbm_test)
        accuracy_rbm = accuracy_score(y_test, predictions_rbm)
        print(f"Exactitud de clasificación usando RBM: {accuracy_rbm}")

        # Evaluar Exactitud de Clasificación usando DBN
        print("\nEvaluando Exactitud de Clasificación usando DBN...")
        features_dbn_train = dbn.transform(x_train)
        features_dbn_test = dbn.transform(x_test)
        classifier_dbn = LogisticRegression(max_iter=1000)
        classifier_dbn.fit(features_dbn_train, y_train)
        predictions_dbn = classifier_dbn.predict(features_dbn_test)
        accuracy_dbn = accuracy_score(y_test, predictions_dbn)
        print(f"Exactitud de clasificación usando DBN: {accuracy_dbn}")

        # Comparar las Exactitudes
        if accuracy_dbn > accuracy_rbm:
            print("\nLa DBN logra una mayor exactitud de clasificación que la RBM individual.")
        else:
            print("\nLa RBM individual logra una mayor exactitud de clasificación que la DBN.")

    # Comparar los MSE finales
    mse_rbm_final = mse_rbm_list[-1] if mse_rbm_list else mean_squared_error(x_test, rbm_individual.reconstruct(x_test))
    mse_dbn_final = mse_dbn_list[-1] if mse_dbn_list else mean_squared_error(x_test, dbn.reconstruct(x_test))
    print(f"\nMSE Final de la RBM individual: {mse_rbm_final}")
    print(f"MSE Final de la DBN: {mse_dbn_final}")

    if mse_dbn_final < mse_rbm_final:
        print("\nLa DBN reconstruye las imágenes con menor error que la RBM individual.")
    else:
        print("\nLa RBM individual reconstruye las imágenes con menor error que la DBN.")

    # Evaluar Exactitud de Clasificación final usando RBM
    print("\nEvaluando Exactitud de Clasificación Final usando RBM...")
    features_rbm_train = rbm_individual.transform(x_train)
    features_rbm_test = rbm_individual.transform(x_test)
    classifier_rbm = LogisticRegression(max_iter=1000)
    classifier_rbm.fit(features_rbm_train, y_train)
    predictions_rbm = classifier_rbm.predict(features_rbm_test)
    accuracy_rbm = accuracy_score(y_test, predictions_rbm)
    print(f"Exactitud de clasificación final usando RBM: {accuracy_rbm}")

    # Evaluar Exactitud de Clasificación final usando DBN
    print("\nEvaluando Exactitud de Clasificación Final usando DBN...")
    features_dbn_train = dbn.transform(x_train)
    features_dbn_test = dbn.transform(x_test)
    classifier_dbn = LogisticRegression(max_iter=1000)
    classifier_dbn.fit(features_dbn_train, y_train)
    predictions_dbn = classifier_dbn.predict(features_dbn_test)
    accuracy_dbn = accuracy_score(y_test, predictions_dbn)
    print(f"Exactitud de clasificación final usando DBN: {accuracy_dbn}")

    # Comparar las Exactitudes finales
    if accuracy_dbn > accuracy_rbm:
        print("\nLa DBN logra una mayor exactitud de clasificación que la RBM individual.")
    else:
        print("\nLa RBM individual logra una mayor exactitud de clasificación que la DBN.")

    # Visualizar los errores por época
    epochs_range = range(1, epochs + 1)  # Rango de épocas

    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, mse_dbn_list, label='DBN', marker='o')
    plt.plot(epochs_range, mse_rbm_list, label='RBM Individual', marker='s')
    plt.xlabel('Época')
    plt.ylabel('Error Cuadrático Medio (MSE)')
    plt.title('Comparación del MSE por Época: DBN vs RBM Individual')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Visualizar algunas reconstrucciones (opcional)
    for i in range(15):
        # Seleccionar una imagen de prueba
        test_image = x_test[i]

        # Agregar ruido a la imagen de prueba
        noisy_image = add_noise_to_image(x_test[i], noise_level=0.07)
        noisy_image_reshaped = noisy_image.reshape(1, -1)

        # Reconstruir con RBM individual
        reconstructed_rbm = rbm_individual.reconstruct(noisy_image_reshaped).flatten()

        # Reconstruir con DBN
        reconstructed_dbn = dbn.reconstruct(noisy_image_reshaped).flatten()

        # Visualizar los resultados
        plt.figure(figsize=(15, 4))

        plt.subplot(1, 4, 1)
        plt.title("Imagen Original")
        plt.imshow(test_image.reshape(28, 28), cmap='gray')
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.title("Imagen con Ruido")
        plt.imshow(noisy_image.reshape(28, 28), cmap='gray')
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.title("Reconstruida RBM")
        plt.imshow(reconstructed_rbm.reshape(28, 28), cmap='gray')
        plt.axis('off')

        plt.subplot(1, 4, 4)
        plt.title("Reconstruida DBN")
        plt.imshow(reconstructed_dbn.reshape(28, 28), cmap='gray')
        plt.axis('off')

        plt.show()