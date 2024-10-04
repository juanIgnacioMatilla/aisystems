import numpy as np

# Función para leer las matrices desde el archivo
def read_digit_matrices(filename):
    with open(filename, 'r') as file:
        content = file.readlines()
    
    # Convertir el contenido en matrices de 7x5
    matrices = []
    for i in range(0, len(content), 7):  # Cada matriz es de 7 líneas
        matrix = []
        for j in range(7):
            # Convertir cada línea en una lista de enteros
            line = list(map(int, content[i + j].strip().split()))
            matrix.append(line)
        matrices.append(np.array(matrix))
    
    return matrices

# Función para generar variaciones con ruido gaussiano
def generate_variations(matrix, alpha, num_variations=10):
    variations = []
    
    for _ in range(num_variations):
        # Generar ruido gaussiano con media 0 y desviación estándar alpha
        noise = np.random.normal(0, alpha, matrix.shape)
        
        # Agregar ruido a la matriz original
        noisy_matrix = matrix + noise
        
        # Asegurarse de que los valores siguen siendo binarios (0 o 1)
        noisy_matrix = np.where(noisy_matrix > 0.5, 1, 0)
        
        variations.append(noisy_matrix)
    
    return variations

# Parámetro de ruido (alpha)
alpha = 0.2

# Leer las matrices del archivo
filename = '../inputs/TP3-ej3-digitos.txt'
digit_matrices = read_digit_matrices(filename)

# Nombre del archivo de salida
output_filename = f'test_noise_{alpha}.txt'

# Guardar todas las variaciones en un archivo
with open(output_filename, 'w') as output_file:
    for index, digit_matrix in enumerate(digit_matrices):
        variations = generate_variations(digit_matrix, alpha)
        
        for i, var in enumerate(variations):
            for row in var:
                output_file.write(' '.join(map(str, row)) + '\n')
print(f"Variaciones guardadas en {output_filename}")