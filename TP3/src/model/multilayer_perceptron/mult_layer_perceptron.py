import numpy as np

from TP3.src.model.multilayer_perceptron.layer import Layer


class MultiLayerPerceptron:
    def __init__(self, layers_structure, learning_rate=0.01, activation_function=lambda x: 1 / (1 + np.exp(-x)),
                 activation_function_derivative=lambda x: x * (1 - x)):
        self.learning_rate = learning_rate
        self.layers = [
            Layer(num_neurons, input_size, activation_function, activation_function_derivative)
            for input_size, num_neurons in zip(layers_structure[:-1], layers_structure[1:])
        ]

    def predict(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer.predict(outputs)
        return outputs

    def train(self, X, y, epochs):
        errors_by_epoch = []
        for epoch in range(epochs):
            epoch_error = 0
            for inputs, target in zip(X, y):
                # Forward propagation
                outputs_by_layer = self.forward_propagate(inputs)

                # Backpropagation
                gradients = self.backward_propagate(outputs_by_layer, target)

                # Update weights
                self.update_weights(inputs, outputs_by_layer, gradients)

                # Error calculation
                final_output = outputs_by_layer[-1]
                epoch_error += 0.5 * np.sum((target - final_output) ** 2)

            errors_by_epoch.append(epoch_error)
        return errors_by_epoch

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
            error = np.zeros(len(layer.neurons))

            # Sumamos los errores de cada neurona en la capa siguiente
            for j, neuron in enumerate(layer.neurons):
                for k, next_neuron in enumerate(next_layer.neurons):
                    # Multiplica el peso de la neurona j con el gradiente de la neurona k
                    error[j] += next_neuron.weights[j] * gradients[-1][k]

            # Calcula el gradiente para la neurona actual
            gradient = error * layer.activation_function_derivative(outputs_by_layer[i + 1])
            gradients.append(gradient)

        return gradients[::-1]  # Invertimos el orden de los gradientes

    def update_weights(self, inputs, outputs_by_layer, gradients):
        for i, layer in enumerate(self.layers):
            layer_inputs = inputs if i == 0 else outputs_by_layer[i]
            layer.adjust_weights(layer_inputs, gradients[i], self.learning_rate)
