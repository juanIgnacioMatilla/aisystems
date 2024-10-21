import numpy as np


class HopfieldNetwork:
    def __init__(self, patterns):
        self.n_neurons = patterns.shape[1]  # Number of neurons equals the dimension of the patterns (5x5 = 25)
        self.patterns = patterns
        self.weights = self._initialize_weights(patterns)

    def _initialize_weights(self, patterns):
        """Initialize weights using the formula W = (1/N) * K * K.T"""
        N = self.n_neurons
        weights = (1 / N) * np.dot(patterns.T, patterns)
        np.fill_diagonal(weights, 0)  # No self-connections
        return weights

    def update(self, states):
        """Actualiza el estado de la red de manera asincrónica."""
        new_states = states.copy()
        for i in range(self.n_neurons):
            # Regla de actualización: Si(t+1) = sgn(sum_j(w_ij * Sj(t)))
            h = np.dot(self.weights[i], states)
            new_states[i] = sgn(h)
        return new_states

    def get_similar(self, pattern, max_iters=100):
        """Run the network until the state no longer changes, track the steps."""
        states = pattern
        states_history = [states.copy()]
        energy_history = [self.energy(states)]
        for i in range(max_iters):
            new_states = self.update(states)
            if np.array_equal(new_states, states):
                break
            states_history.append(new_states)
            energy_history.append(self.energy(new_states))
            states = new_states
        return states, states_history, energy_history

    def energy(self, states):
        """Compute the energy of the current state."""
        return -0.5 * np.dot(states.T, np.dot(self.weights, states))


def sgn(x):
    if x > 0:
        return 1
    if x < 0:
        return -1
    else:
        return x
