import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)  # x is the output of sigmoid


class Autoencoder:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)

        # Initialize weights and biases
        self.weights = []
        self.biases = []

        # np.random.seed(42)
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

    # Activation functions

    def forward(self, x):
        activations = [x]
        zs = []
        for i in range(self.num_layers - 1):
            z = np.dot(self.weights[i], activations[-1]) + self.biases[i]
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        return activations, zs

    def backward(self, x, activations, zs):
        grads_w = [np.zeros_like(w) for w in self.weights]
        grads_b = [np.zeros_like(b) for b in self.biases]

        # Calculate the error at the output layer
        delta = (activations[-1] - x) * sigmoid_derivative(activations[-1])
        grads_w[-1] = np.dot(delta, activations[-2].T)
        grads_b[-1] = delta

        # Backpropagate the error
        for l in range(2, self.num_layers):
            z = zs[-l]
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

    def train(self, X, learning_rate=0.001, num_epochs=4500, error_bar_interval=500):
        """
        Trains the autoencoder.

        Args:
            X (np.ndarray): Input data of shape (num_samples, input_size).
            learning_rate (float): Learning rate for weight updates.
            num_epochs (int): Number of training epochs.
            error_bar_interval (int): Interval (in epochs) to record max pixel error.

        Returns:
            loss_history (list): List of average loss per epoch.
            max_pixel_error_history (list): List of max pixel errors at specified intervals.
        """
        loss_history = []
        max_pixel_error_history = []  # To store maximum pixel error per interval
        for epoch in range(1, num_epochs + 1):
            total_loss = 0
            for i in range(X.shape[0]):
                x = X[i].reshape(-1, 1)
                activations, zs = self.forward(x)
                loss = np.mean((activations[-1] - x) ** 2)
                total_loss += loss
                grads_w, grads_b = self.backward(x, activations, zs)
                self.update_parameters(grads_w, grads_b, learning_rate)
            avg_loss = total_loss / X.shape[0]
            loss_history.append(avg_loss)

            # Compute and store the max pixel error at specified intervals
            if epoch % error_bar_interval == 0:
                pixel_errors = []
                for i in range(X.shape[0]):
                    x = X[i].reshape(-1, 1)
                    activations, _ = self.forward(x)
                    output = activations[-1]
                    reconstructed = (output > 0.5).astype(int)
                    pixel_error = np.sum(np.abs(reconstructed - x))
                    pixel_errors.append(pixel_error)
                max_pixel_error = max(pixel_errors)
                max_pixel_error_history.append(max_pixel_error)
                print(f"Epoch {epoch}/{num_epochs}, Average Loss: {avg_loss:.6f}, Max Pixel Error: {max_pixel_error}")
        return loss_history, max_pixel_error_history

    def reconstruct(self, x):
        activations, _ = self.forward(x)
        return activations[-1]

    def decode(self, latent_vector):
        """
        Decode a latent vector to generate a new character.

        Args:
            latent_vector (numpy.ndarray): A column vector of shape (latent_size, 1).

        Returns:
            numpy.ndarray: The reconstructed output vector.
        """
        activation = latent_vector
        # Pass through decoder layers (assuming symmetric architecture)
        # The decoder is the second half of the weights
        for i in range(len(self.weights) // 2, len(self.weights)):
            z = np.dot(self.weights[i], activation) + self.biases[i]
            activation = sigmoid(z)
        return activation
