# Ejemplo de uso
import random

import numpy as np

from TP4.src.model.hopfield.hopfield_network import HopfieldNetwork


def add_noise(pattern, noise_level=0.1):
    """Add noise to a pattern by flipping a percentage of bits."""
    noisy_pattern = pattern.copy()
    num_flips = int(noise_level * pattern.size)  # Calculate how many bits to flip
    indices = random.sample(range(pattern.size), num_flips)
    for idx in indices:
        noisy_pattern[idx] *= -1  # Flip the bit
    return noisy_pattern


def print_pattern(pattern, shape=(5, 5)):
    """Print a 1D pattern as a 2D matrix (5x5)."""
    pattern_reshaped = pattern.reshape(shape)
    for row in pattern_reshaped:
        print(' '.join('*' if x == 1 else ' ' for x in row))
    print("\n")


if __name__ == "__main__":
    # Define the letter J pattern
    letter_J = np.array([
         1,  1,  1,  1,  1,
        -1, -1, -1,  1, -1,
        -1, -1, -1,  1, -1,
        -1,  -1, -1,  1, -1,
         1,  1,  1, 1, -1
    ])

    # Define the letter A pattern
    letter_A = np.array([
        1,  1,  1,  1, 1,
        1, -1, -1, -1, 1,
        1,  1,  1,  1, 1,
        1, -1, -1, -1, 1,
        1, -1, -1, -1, 1,
    ])

    # Define the letter E pattern
    letter_E = np.array([
        1,  1, 1, 1, 1,
        1, -1, -1, -1, -1,
        1, 1, 1, 1, 1,
        1, -1, -1, -1, -1,
        1, 1, 1, 1, 1
    ])

    # Define the letter L pattern
    letter_L = np.array([
        1, -1, -1, -1, -1,
        1, -1, -1, -1, -1,
        1, -1, -1, -1, -1,
        1, -1, -1, -1, -1,
        1, 1, 1, 1, 1
    ])

    # Stack patterns to form the pattern matrix
    patterns = np.vstack([letter_J, letter_A, letter_E, letter_L])

    # Create the Hopfield network with the patterns
    hopfield_net = HopfieldNetwork(patterns)

    # Define a noisy version of one of the patterns
    noisy_letter = add_noise(letter_A, noise_level=0.1)

    # Print the noisy pattern
    print("Noisy pattern:")
    print_pattern(noisy_letter)

    # Recover the pattern by running the network until convergence
    recovered_pattern, states, energies = hopfield_net.get_similar(noisy_letter.copy(), 100)

    # Print all the steps leading to the final state
    print("Steps towards convergence:")
    for i, state in enumerate(states):
        print(f"Step {i + 1}:")
        print_pattern(state)

    print("Energy across steps:")
    for i, energy in enumerate(energies):
        print(f"Energy in step {i+1}: {energy:.3f}")