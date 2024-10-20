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
        """Update the states using matrix multiplication."""
        for i in range(self.n_neurons):
            # Calcular h_i
            h_i = np.dot(self.weights[i], states)
            # Actualizar S_i de acuerdo a la regla:
            if h_i > 0:
                states[i] = 1
            elif h_i < 0:
                states[i] = -1
            # Si h_i = 0, el estado de la neurona no cambia (se mantiene prev_state[i])
        return np.array(states)

    def get_similar(self, pattern, max_iters=100):
            """Run the network until the state no longer changes, track the steps."""
            states = pattern
            states_history = [states.copy()]
            energy_history = [self.energy(states)]
            for i in range(max_iters):
                new_states = self.update(states)
                states_history.append(new_states.copy())
                energy_history.append(self.energy(new_states))
                if np.array_equal(new_states, states):
                    break
                states = new_states

            return states, states_history, energy_history

    def energy(self, states):
        """Compute the energy of the current state."""
        return -0.5 * np.dot(states.T, np.dot(self.weights, states))
