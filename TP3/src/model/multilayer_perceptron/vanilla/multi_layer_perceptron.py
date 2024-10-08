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

    def train(self, X, y, epochs, batch_size=None, online=False):
        # time to train
        start_time = time.time()

        n_samples = X.shape[0]

        # Handle batch size:
        if online:
            batch_size = 1  # Online learning (SGD)
        elif batch_size is None:
            batch_size = n_samples  # Full-batch if no batch size is provided

        for epoch in range(epochs):

            # Shuffle data at the beginning of each epoch
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X, y = X[indices], y[indices]

            epoch_error = 0
            correct_predictions = 0

            # Process data in batches
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_X = X[start_idx:end_idx]
                batch_y = y[start_idx:end_idx]

                batch_error = 0
                batch_correct_predictions = 0

                # Forward propagation for the batch
                batch_outputs_by_layer = []
                for inputs in batch_X:
                    outputs_by_layer = self.forward_propagate(inputs)
                    batch_outputs_by_layer.append(outputs_by_layer)

                # Backpropagation for the batch
                batch_gradients = [self.backward_propagate(outputs, target)
                                   for outputs, target in zip(batch_outputs_by_layer, batch_y)]

                # Update weights for the batch
                for inputs, outputs_by_layer, gradients, target in zip(batch_X, batch_outputs_by_layer, batch_gradients,
                                                                       batch_y):
                    self.update_weights(inputs, outputs_by_layer, gradients)

                    # Error calculation
                    final_output = outputs_by_layer[-1]
                    batch_error += 0.5 * np.sum((target - final_output) ** 2)

                    # Accuracy calculation
                    predicted_class = np.argmax(final_output)
                    true_class = np.argmax(target)
                    if predicted_class == true_class:
                        batch_correct_predictions += 1

                # Aggregate batch error and accuracy
                epoch_error += batch_error
                correct_predictions += batch_correct_predictions

            # Mean error and accuracy for this epoch
            mean_error = epoch_error / n_samples
            accuracy = correct_predictions / n_samples

            self.errors_by_epoch.append(mean_error)
            self.accuracies_by_epoch.append(accuracy)

        # time to train
        self.training_time = time.time() - start_time
        return self.errors_by_epoch, self.accuracies_by_epoch

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
