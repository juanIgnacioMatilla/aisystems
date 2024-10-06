import time

import numpy as np

from TP3.src.model.multilayer_perceptron.vanilla.layer import Layer

class MultiLayerPerceptron:
    def __init__(self, layers_structure, learning_rate=0.01, activation_function=lambda x: 1 / (1 + np.exp(-x)),
                 activation_function_derivative=lambda x: x * (1 - x)):
        self.errors_by_epoch = []
        self.accuracies_by_epoch = []
        self.training_time = 0
        self.learning_rate = learning_rate
        self.layers = [
            Layer(num_neurons, input_size, activation_function, activation_function_derivative)
            for input_size, num_neurons in zip(layers_structure[:-1], layers_structure[1:])
        ]

    def __getstate__(self):
        # Return the state of the object to be pickled
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        # Restore the state of the object from the unpickled state
        self.__dict__.update(state)

    def predict(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer.predict(outputs)
        return outputs

    F1 - SCORE

    def train(self, X, y, epochs):
        # Time to train
        start_time = time.time()

        f1_by_epoch = []

        for epoch in range(epochs):
            epoch_error = 0
            correct_predictions = 0
            confusion_matrix = np.zeros((10, 10))

            for inputs, target in zip(X, y):
                # Forward propagation
                outputs_by_layer = self.forward_propagate(inputs)

                # Backpropagation
                gradients = self.backward_propagate(outputs_by_layer, target)

                # Update weights
                self.update_weights(inputs, outputs_by_layer, gradients)

                # Error calculation (Mean Squared Error)
                final_output = outputs_by_layer[-1]
                epoch_error += 0.5 * np.sum((target - final_output) ** 2)

                # Accuracy calculation (assuming target is one-hot encoded)
                predicted_class = np.argmax(final_output)
                true_class = np.argmax(target)
                if predicted_class == true_class:
                    correct_predictions += 1

                # Update confusion matrix
                confusion_matrix[true_class][predicted_class] += 1

            # Calculate mean error and accuracy for this epoch
            mean_error = epoch_error / len(X)
            accuracy = correct_predictions / len(X)

            self.errors_by_epoch.append(mean_error)
            self.accuracies_by_epoch.append(accuracy)

            # Calculate F1 score for each class
            f1_scores = []
            for i in range(10):
                tp = confusion_matrix[i][i]
                fn = sum([confusion_matrix[i][j] for j in range(10) if j != i])
                fp = sum([confusion_matrix[j][i] for j in range(10) if j != i])
                precision = tp / (tp + fp) if tp + fp != 0 else 0
                recall = tp / (tp + fn) if tp + fn != 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
                f1_scores.append(f1)

            # Store the average F1 score for this epoch
            f1_by_epoch.append(np.mean(f1_scores))

        # Time to train
        self.training_time = time.time() - start_time
        return self.errors_by_epoch, self.accuracies_by_epoch, f1_by_epoch

    def forward_propagate(self, inputs):
        outputs_by_layer = [inputs]
        for layer in self.layers:
            outputs = layer.predict(outputs_by_layer[-1])
            outputs_by_layer.append(outputs)
        return outputs_by_layer

    def backward_propagate(self, outputs_by_layer, target):
        gradients = []

        # Error en la capa de salida
        output_layer_index = -1
        error = target - outputs_by_layer[output_layer_index]
        gradient = error * self.layers[output_layer_index].activation_function_derivative(
            outputs_by_layer[output_layer_index])
        gradients.append(gradient)

        # Retropropagación en capas ocultas
        for i in reversed(range(len(self.layers) - 1)):
            layer = self.layers[i]
            next_layer = self.layers[i + 1]
            # Transponer la matriz de pesos
            weights_transposed = np.transpose([neuron.weights[:-1] for neuron in next_layer.neurons])
            # Multiplicar los gradientes de la capa siguiente por los pesos transpuestos
            error = np.dot(weights_transposed, gradients[-1])

            # Calcula el gradiente para la neurona actual
            gradient = error * layer.activation_function_derivative(outputs_by_layer[i + 1])
            gradients.append(gradient)

        return gradients[::-1]  # Invertimos el orden de los gradientes

    def update_weights(self, inputs, outputs_by_layer, gradients):
        for i, layer in enumerate(self.layers):
            layer_inputs = inputs if i == 0 else outputs_by_layer[i]
            layer.adjust_weights(layer_inputs, gradients[i], self.learning_rate)
