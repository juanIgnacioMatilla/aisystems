import json
import sys
import numpy as np

from TP4.hopfield_tests.main_hopfield import add_noise
from TP4.src.model.hopfield import letters
from TP4.src.model.hopfield.hopfield_network import HopfieldNetwork


def main(config):

    # Validar y extraer parámetros
    try:
        pattern_names = config['pattern_names']
        input_pattern_name = config['input_pattern_name']
        max_iters = config.get('max_iters', 100)  # Valor por defecto 100 si no se especifica
    except KeyError as e:
        print(f"Falta el parámetro en el archivo de configuración: {e}")
        sys.exit(1)

    # Obtener los patrones desde letters.py
    patterns = []
    for name in pattern_names:
        try:
            pattern = getattr(letters, name)
            if pattern.shape[0] != 25:
                print(f"Error: El patrón '{name}' no tiene 25 elementos (5x5).")
                sys.exit(1)
            patterns.append(pattern)
        except AttributeError:
            print(f"Error: '{name}' no está definido en 'letters.py'.")
            sys.exit(1)
        except Exception as e:
            print(f"Error al obtener el patrón '{name}': {e}")
            sys.exit(1)

    # Convertir la lista de patrones en una matriz 2D
    patterns = np.array(patterns)

    # Obtener el patrón de entrada
    try:
        input_pattern = getattr(letters, input_pattern_name)
        if input_pattern.shape[0] != 25:
            print(f"Error: El patrón de entrada '{input_pattern_name}' no tiene 25 elementos (5x5).")
            sys.exit(1)
    except AttributeError:
        print(f"Error: '{input_pattern_name}' no está definido en 'letters.py'.")
        sys.exit(1)
    except Exception as e:
        print(f"Error al obtener el patrón de entrada '{input_pattern_name}': {e}")
        sys.exit(1)

    for pattern in patterns:
        print_pattern(pattern)
    # Create the Hopfield network with the patterns
    hopfield_net = HopfieldNetwork(patterns)

    # Define a noisy version of one of the patterns
    noisy_letter = add_noise(patterns[1], noise_level=0.2)

    # Print the noisy pattern
    print("Noisy pattern:")
    print_pattern(noisy_letter)

    # Recover the pattern by running the network until convergence
    recovered_pattern, states_history, energies = hopfield_net.get_similar(noisy_letter.copy(), 100)

    # Print all the steps leading to the final state
    print("Steps towards convergence:")
    for i, states in enumerate(states_history):
        print(f"Step {i + 1}:")
        print_pattern(states)

    print("Energy across steps:")
    for i, energy in enumerate(energies):
        print(f"Energy in step {i + 1}: {energy:.3f}")


def load_config(filename: str):
    """Load configuration from a JSON file."""
    with open(filename, 'r') as file:
        return json.load(file)

def print_pattern(pattern, shape=(5, 5)):
    """Print a 1D pattern as a 2D matrix (5x5)."""
    pattern_reshaped = pattern.reshape(shape)
    for row in pattern_reshaped:
        print(' '.join('*' if x == 1 else ' ' for x in row))
    print("\n")

if __name__ == "__main__":
    # Load configuration from JSON
    config = load_config('config.json')
    main(config)
