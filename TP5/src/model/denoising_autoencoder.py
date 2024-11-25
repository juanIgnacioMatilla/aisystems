import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(output):
    return output * (1 - output)  # 'output' is the sigmoid activation

class DenoisingAutoencoder:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)

        # Initialize weights and biases
        self.weights = []
        self.biases = []

        for i in range(self.num_layers - 1):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i + 1]
            # Xavier Initialization
            limit = np.sqrt(6 / (n_in + n_out))
            weight = np.random.uniform(-limit, limit, (n_out, n_in))
            bias = np.zeros((n_out, 1))
            self.weights.append(weight)
            self.biases.append(bias)

        # Initialize Adam parameters
        self.m_weights = [np.zeros_like(w) for w in self.weights]
        self.v_weights = [np.zeros_like(w) for w in self.weights]
        self.m_biases = [np.zeros_like(b) for b in self.biases]
        self.v_biases = [np.zeros_like(b) for b in self.biases]
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0  # Time step

    def forward(self, x):
        activations = [x]
        zs = []
        for i in range(self.num_layers - 1):
            z = np.dot(self.weights[i], activations[-1]) + self.biases[i]
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        return activations, zs

    def backward(self, x_target, activations, zs):
        grads_w = [np.zeros_like(w) for w in self.weights]
        grads_b = [np.zeros_like(b) for b in self.biases]

        # Calculate the error at the output layer
        delta = (activations[-1] - x_target) * sigmoid_derivative(activations[-1])
        grads_w[-1] = np.dot(delta, activations[-2].T)
        grads_b[-1] = delta

        # Backpropagate the error
        for l in range(2, self.num_layers):
            sp = sigmoid_derivative(activations[-l])
            delta = np.dot(self.weights[-l + 1].T, delta) * sp
            grads_w[-l] = np.dot(delta, activations[-l - 1].T)
            grads_b[-l] = delta

        return grads_w, grads_b

    def update_parameters(self, grads_w, grads_b, learning_rate):
        self.t += 1
        for i in range(len(self.weights)):
            # Update biased first moment estimate
            self.m_weights[i] = self.beta1 * self.m_weights[i] + (1 - self.beta1) * grads_w[i]
            self.m_biases[i] = self.beta1 * self.m_biases[i] + (1 - self.beta1) * grads_b[i]
            # Update biased second raw moment estimate
            self.v_weights[i] = self.beta2 * self.v_weights[i] + (1 - self.beta2) * (grads_w[i] ** 2)
            self.v_biases[i] = self.beta2 * self.v_biases[i] + (1 - self.beta2) * (grads_b[i] ** 2)
            # Compute bias-corrected first moment estimate
            m_hat_w = self.m_weights[i] / (1 - self.beta1 ** self.t)
            m_hat_b = self.m_biases[i] / (1 - self.beta1 ** self.t)
            # Compute bias-corrected second raw moment estimate
            v_hat_w = self.v_weights[i] / (1 - self.beta2 ** self.t)
            v_hat_b = self.v_biases[i] / (1 - self.beta2 ** self.t)
            # Update parameters
            self.weights[i] -= learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
            self.biases[i] -= learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

    def train(self, X_noisy, X_clean, learning_rate=0.001, num_epochs=5000):
        loss_history = []
        num_samples = X_noisy.shape[0]
        for epoch in range(num_epochs):
            total_loss = 0
            for i in range(num_samples):
                x_noisy = X_noisy[i].reshape(-1, 1)
                x_clean = X_clean[i].reshape(-1, 1)
                activations, zs = self.forward(x_noisy)
                loss = np.mean((activations[-1] - x_clean) ** 2)
                total_loss += loss
                grads_w, grads_b = self.backward(x_clean, activations, zs)
                self.update_parameters(grads_w, grads_b, learning_rate)
            avg_loss = total_loss / num_samples
            loss_history.append(avg_loss)
            if (epoch + 1) % 500 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss}")
        return loss_history

    def reconstruct(self, x):
        activations, _ = self.forward(x)
        return activations[-1]

    def encode(self, x):
        """
        Encode the input vector into the latent space.
        Args:
            x (numpy.ndarray): A column vector of shape (input_size, 1)
        Returns:
            numpy.ndarray: The latent representation
        """
        activation = x
        # Pass through encoder layers
        num_encoder_layers = len(self.weights) // 2
        for i in range(num_encoder_layers):
            z = np.dot(self.weights[i], activation) + self.biases[i]
            activation = sigmoid(z)
        return activation

    def decode(self, latent_vector):
        """
        Decode a latent vector to generate a new character.
        Args:
            latent_vector (numpy.ndarray): A column vector of shape (latent_size, 1)
        Returns:
            numpy.ndarray: The reconstructed output vector
        """
        activation = latent_vector
        # Pass through decoder layers
        for i in range(len(self.weights) // 2, len(self.weights)):
            z = np.dot(self.weights[i], activation) + self.biases[i]
            activation = sigmoid(z)
        return activation
