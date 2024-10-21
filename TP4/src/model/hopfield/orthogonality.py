import numpy as np
import pandas as pd

from TP4.src.model.hopfield.letters import letter_A, letter_B, letter_C, letter_D, letter_E, letter_F, letter_G, \
    letter_H, letter_I, letter_J, letter_K, letter_L, letter_M, letter_N, letter_O, letter_P, letter_Q, letter_R, \
    letter_S, letter_T, letter_U, letter_V, letter_W, letter_X, letter_Y, letter_Z

# Lista de todos los patrones
patterns = {
    'A': letter_A,
    'B': letter_B,
    'C': letter_C,
    'D': letter_D,
    'E': letter_E,
    'F': letter_F,
    'G': letter_G,
    'H': letter_H,
    'I': letter_I,
    'J': letter_J,
    'K': letter_K,
    'L': letter_L,
    'M': letter_M,
    'N': letter_N,
    'O': letter_O,
    'P': letter_P,
    'Q': letter_Q,
    'R': letter_R,
    'S': letter_S,
    'T': letter_T,
    'U': letter_U,
    'V': letter_V,
    'W': letter_W,
    'X': letter_X,
    'Y': letter_Y,
    'Z': letter_Z
}


# Calcular el producto escalar entre cada par de letras
def calculate_orthogonality(patterns):
    letters = list(patterns.keys())
    n = len(letters)

    # Matriz para almacenar los productos escalares
    product_matrix = np.zeros((n, n))

    # Calcular el producto escalar entre todos los pares
    for i in range(n):
        for j in range(i, n):  # Para no repetir cálculos (ya que es simétrica)
            product = np.dot(patterns[letters[i]], patterns[letters[j]])
            product_matrix[i, j] = product
            product_matrix[j, i] = product  # Simetría

    return letters, product_matrix


# Función para encontrar letras que formen un conjunto ortogonal
def find_orthogonal_subset(min_orthogonality, max_orthogonality, subset_size=4):
    letters, product_matrix = calculate_orthogonality(patterns)
    n = len(letters)
    selected_letters = []

    # Probar todas las combinaciones posibles de 4 letras
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    group = [letters[i], letters[j], letters[k], letters[l]]
                    valid = True

                    # Verificar que todas las combinaciones entre las letras del grupo respeten la ortogonalidad
                    for x in range(subset_size):
                        for y in range(x + 1, subset_size):
                            orthogonality = abs(product_matrix[letters.index(group[x])][letters.index(group[y])])
                            if not (min_orthogonality < orthogonality < max_orthogonality):
                                valid = False
                                break
                        if not valid:
                            break

                    if valid:
                        return get_matrix_of_letters(group)  # Devolvemos el primer conjunto que cumple la condición
    return None  # Si no se encuentra ningún conjunto, devolvemos una lista vacía


def get_matrix_of_letters(letters):
    letters_matrix = []
    for letter in letters:
        letters_matrix.append(patterns[letter])
    return np.vstack(letters_matrix)


def print_pattern(pattern, shape=(5, 5)):
    """Print a 1D pattern as a 2D matrix (5x5)."""
    pattern_reshaped = pattern.reshape(shape)
    for row in pattern_reshaped:
        print(' '.join('*' if x == 1 else ' ' for x in row))
    print("\n")
