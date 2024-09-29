import numpy as np

from TP3.src.model.multilayer_perceptron.vanilla.layer import Layer


class AdamLayer(Layer):
    def __init__(self, num_neurons, input_size, activation_function, activation_function_derivative, beta1,beta2,epsilon):
        super().__init__(num_neurons, input_size, activation_function, activation_function_derivative)
        # Variables para Adam
        self.m_t = [np.zeros_like(neuron.weights) for neuron in self.neurons]  # Primer momento
        self.v_t = [np.zeros_like(neuron.weights) for neuron in self.neurons]  # Segundo momento
        self.t = 0  # Contador de pasos
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def predict(self, inputs):
        return np.array([neuron.predict(inputs) for neuron in self.neurons])

    def adjust_weights(self, inputs, deltas, learning_rate):
        inputs_with_bias = np.append(inputs, 1)  # Incluimos el término de sesgo
        self.t += 1  # Incrementamos el contador de pasos

        for i, (neuron, delta) in enumerate(zip(self.neurons, deltas)):
            # Actualizamos el primer momento (m_t) y el segundo momento (v_t)
            self.m_t[i] = self.beta1 * self.m_t[i] + (1 - self.beta1) * delta * inputs_with_bias
            self.v_t[i] = self.beta2 * self.v_t[i] + (1 - self.beta2) * (delta * inputs_with_bias) ** 2

            # Aplicamos las correcciones de sesgo
            m_hat = self.m_t[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v_t[i] / (1 - self.beta2 ** self.t)

            # Actualizamos los pesos utilizando Adam
            neuron.weights += learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
