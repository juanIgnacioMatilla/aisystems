import numpy as np


class VariationalAutoencoder:
    def __init__(self, input_size, hidden_sizes, latent_size):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.latent_size = latent_size

        # Initialize weights and biases
        self.weights = {}
        self.biases = {}
        self.initialize_weights()

        # Initialize Adam parameters
        self.m_weights = {key: np.zeros_like(val) for key, val in self.weights.items()}
        self.v_weights = {key: np.zeros_like(val) for key, val in self.weights.items()}
        self.m_biases = {key: np.zeros_like(val) for key, val in self.biases.items()}
        self.v_biases = {key: np.zeros_like(val) for key, val in self.biases.items()}
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0  # Time step
        # Initialize loss history
        self.loss_history = []

    def initialize_weights(self):
        np.random.seed(42)
        # Encoder weights
        encoder_layer_sizes = [self.input_size] + self.hidden_sizes
        for i in range(len(encoder_layer_sizes) - 1):
            n_in = encoder_layer_sizes[i]
            n_out = encoder_layer_sizes[i + 1]
            limit = np.sqrt(6 / (n_in + n_out))
            self.weights[f'encoder_W{i + 1}'] = np.random.uniform(-limit, limit, (n_out, n_in))
            self.biases[f'encoder_b{i + 1}'] = np.zeros((n_out, 1))
        # Mean and log variance weights
        n_last = self.hidden_sizes[-1]
        limit = np.sqrt(6 / (n_last + self.latent_size))
        self.weights['mu_W'] = np.random.uniform(-limit, limit, (self.latent_size, n_last))
        self.biases['mu_b'] = np.zeros((self.latent_size, 1))
        self.weights['logvar_W'] = np.random.uniform(-limit, limit, (self.latent_size, n_last))
        self.biases['logvar_b'] = np.zeros((self.latent_size, 1))
        # Decoder weights
        decoder_layer_sizes = [self.latent_size] + self.hidden_sizes[::-1]
        for i in range(len(decoder_layer_sizes) - 1):
            n_in = decoder_layer_sizes[i]
            n_out = decoder_layer_sizes[i + 1]
            limit = np.sqrt(6 / (n_in + n_out))
            self.weights[f'decoder_W{i + 1}'] = np.random.uniform(-limit, limit, (n_out, n_in))
            self.biases[f'decoder_b{i + 1}'] = np.zeros((n_out, 1))
        # Output layer weights
        n_in = self.hidden_sizes[0]
        n_out = self.input_size
        limit = np.sqrt(6 / (n_in + n_out))
        self.weights['decoder_W_out'] = np.random.uniform(-limit, limit, (n_out, n_in))
        self.biases['decoder_b_out'] = np.zeros((n_out, 1))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        activations = {'a0': x}
        # Encoder forward pass
        a = x
        for i in range(len(self.hidden_sizes)):
            W = self.weights[f'encoder_W{i+1}']
            b = self.biases[f'encoder_b{i+1}']
            z = np.dot(W, a) + b
            a = self.relu(z)
            activations[f'z_enc{i+1}'] = z
            activations[f'a_enc{i+1}'] = a
        # Latent space
        W_mu = self.weights['mu_W']
        b_mu = self.biases['mu_b']
        W_logvar = self.weights['logvar_W']
        b_logvar = self.biases['logvar_b']
        mu = np.dot(W_mu, a) + b_mu
        logvar = np.dot(W_logvar, a) + b_logvar
        activations['mu'] = mu
        activations['logvar'] = logvar
        # Reparameterization trick
        std = np.exp(0.5 * logvar)
        epsilon = np.random.randn(*std.shape)
        z = mu + std * epsilon
        activations['z'] = z
        # Decoder forward pass
        a = z
        for i in range(len(self.hidden_sizes)):
            W = self.weights[f'decoder_W{i+1}']
            b = self.biases[f'decoder_b{i+1}']
            z_dec = np.dot(W, a) + b
            a = self.relu(z_dec)
            activations[f'z_dec{i+1}'] = z_dec
            activations[f'a_dec{i+1}'] = a
        # Output layer
        W_out = self.weights['decoder_W_out']
        b_out = self.biases['decoder_b_out']
        y = np.dot(W_out, a) + b_out
        y = self.sigmoid(y)
        activations['y'] = y
        return activations

    def compute_loss(self, x, activations):
        y = activations['y']
        mu = activations['mu']
        logvar = activations['logvar']
        # Reconstruction loss (binary cross-entropy)
        bce = -np.sum(x * np.log(y + 1e-9) + (1 - x) * np.log(1 - y + 1e-9))
        # KL divergence
        kl = -0.5 * np.sum(1 + logvar - mu ** 2 - np.exp(logvar))
        return bce + kl

    def backward(self, x, activations):
        grads = {}
        y = activations['y']
        mu = activations['mu']
        logvar = activations['logvar']
        z = activations['z']

        # Reconstruction loss gradient
        dL_dy = -(x / (y + 1e-9)) + ((1 - x) / (1 - y + 1e-9))
        dy_dz = y * (1 - y)
        delta = dL_dy * dy_dz  # Shape: (input_size, 1)

        # Output layer gradients
        a_prev = activations[f'a_dec{len(self.hidden_sizes)}']
        grads['decoder_W_out'] = np.dot(delta, a_prev.T)
        grads['decoder_b_out'] = delta

        # Backpropagate through decoder hidden layers
        delta = np.dot(self.weights['decoder_W_out'].T, delta)
        for i in reversed(range(len(self.hidden_sizes))):
            z_dec = activations[f'z_dec{i + 1}']
            dz = self.relu_derivative(z_dec)
            delta *= dz
            a_prev = activations['z'] if i == 0 else activations[f'a_dec{i}']
            grads[f'decoder_W{i + 1}'] = np.dot(delta, a_prev.T)
            grads[f'decoder_b{i + 1}'] = delta
            # Update delta in every iteration
            delta = np.dot(self.weights[f'decoder_W{i + 1}'].T, delta)

        # At this point, delta should have the shape (latent_size, 1)
        dL_dz = delta
        # KL Divergence gradients
        dL_dmu = dL_dz + mu
        dL_dlogvar = 0.5 * (dL_dz * z - dL_dz * mu - 1 + np.exp(logvar))

        # Gradients for mu and logvar weights
        a_enc_last = activations[f'a_enc{len(self.hidden_sizes)}']
        grads['mu_W'] = np.dot(dL_dmu, a_enc_last.T)
        grads['mu_b'] = dL_dmu
        grads['logvar_W'] = np.dot(dL_dlogvar, a_enc_last.T)
        grads['logvar_b'] = dL_dlogvar

        # Backpropagate through encoder hidden layers
        delta = (np.dot(self.weights['mu_W'].T, dL_dmu) +
                 np.dot(self.weights['logvar_W'].T, dL_dlogvar))
        for i in reversed(range(len(self.hidden_sizes))):
            z_enc = activations[f'z_enc{i + 1}']
            dz = self.relu_derivative(z_enc)
            delta *= dz
            a_prev = x if i == 0 else activations[f'a_enc{i}']
            grads[f'encoder_W{i + 1}'] = np.dot(delta, a_prev.T)
            grads[f'encoder_b{i + 1}'] = delta
            if i != 0:
                delta = np.dot(self.weights[f'encoder_W{i + 1}'].T, delta)

        return grads

    def update_parameters(self, grads, learning_rate):
        self.t += 1  # Increment time step
        for key in self.weights:
            # Update biased first moment estimate
            self.m_weights[key] = self.beta1 * self.m_weights[key] + (1 - self.beta1) * grads[key]
            # Update biased second raw moment estimate
            self.v_weights[key] = self.beta2 * self.v_weights[key] + (1 - self.beta2) * (grads[key] ** 2)
            # Compute bias-corrected first moment estimate
            m_hat_w = self.m_weights[key] / (1 - self.beta1 ** self.t)
            # Compute bias-corrected second raw moment estimate
            v_hat_w = self.v_weights[key] / (1 - self.beta2 ** self.t)
            # Update weights
            self.weights[key] -= learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)

        for key in self.biases:
            # Update biased first moment estimate
            self.m_biases[key] = self.beta1 * self.m_biases[key] + (1 - self.beta1) * grads[key]
            # Update biased second raw moment estimate
            self.v_biases[key] = self.beta2 * self.v_biases[key] + (1 - self.beta2) * (grads[key] ** 2)
            # Compute bias-corrected first moment estimate
            m_hat_b = self.m_biases[key] / (1 - self.beta1 ** self.t)
            # Compute bias-corrected second raw moment estimate
            v_hat_b = self.v_biases[key] / (1 - self.beta2 ** self.t)
            # Update biases
            self.biases[key] -= learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

    def train(self, X, batch_size=64, epochs=50, learning_rate=0.001):
        num_samples = X.shape[0]
        num_batches = num_samples // batch_size
        for epoch in range(epochs):
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            epoch_loss = 0
            for i in range(num_batches):
                batch_X = X_shuffled[i * batch_size:(i + 1) * batch_size]
                batch_loss = 0
                grads_accum = {key: np.zeros_like(val) for key, val in self.weights.items()}
                grads_accum.update({key: np.zeros_like(val) for key, val in self.biases.items()})
                for x in batch_X:
                    x = x.reshape(-1, 1)
                    activations = self.forward(x)
                    loss = self.compute_loss(x, activations)
                    batch_loss += loss
                    grads = self.backward(x, activations)
                    for key in grads_accum:
                        grads_accum[key] += grads[key]
                # Average gradients
                for key in grads_accum:
                    grads_accum[key] /= batch_size
                # Update parameters using Adam
                self.update_parameters(grads_accum, learning_rate)
                epoch_loss += batch_loss / batch_size
            avg_loss = epoch_loss / num_batches
            self.loss_history.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    def reconstruct(self, x):
        activations = self.forward(x)
        return activations['y']

    def encode(self, x):
        activations = self.forward(x)
        return activations['mu'], activations['logvar']

    def generate(self, z):
        a = z
        for i in range(len(self.hidden_sizes)):
            W = self.weights[f'decoder_W{i+1}']
            b = self.biases[f'decoder_b{i+1}']
            z_dec = np.dot(W, a) + b
            a = self.relu(z_dec)
        # Output layer
        W_out = self.weights['decoder_W_out']
        b_out = self.biases['decoder_b_out']
        y = np.dot(W_out, a) + b_out
        y = self.sigmoid(y)
        return y

    def generate_grid(self, n=15, digit_size=20):
        """
        Generate a grid of latent variables and decode them to images.

        Args:
            n (int): Number of points per dimension.
            digit_size (int): Size of each digit image in pixels.

        Returns:
            np.ndarray: Combined grid image.
        """
        grid_x = np.linspace(-3, 3, n)
        grid_y = np.linspace(-3, 3, n)
        grid_image = np.zeros((digit_size * n, digit_size * n))
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z = np.array([[xi], [yi]])
                y = self.generate(z)
                digit = y.reshape(digit_size, digit_size)
                grid_image[i * digit_size:(i + 1) * digit_size,
                j * digit_size:(j + 1) * digit_size] = digit
        return grid_image
