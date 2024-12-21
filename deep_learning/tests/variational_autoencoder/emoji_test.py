import os
import random
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm
import json

from deep_learning.src.model.variational_autoencoder import VariationalAutoencoder

# Cargar parámetros desde config.json
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# Obtener parámetros desde config.json
FONT_PATH = config['font_path']
EMOJIS = config['emojis']
IMAGE_SIZE = config['image_size']
SAMPLES_PER_EMOJI = config['samples_per_emoji']
INPUT_SIZE = IMAGE_SIZE ** 2
HIDDEN_SIZES = config['hidden_sizes']
LATENT_SIZE = config['latent_size']
BATCH_SIZE = config['batch_size']
EPOCHS = config['epochs']
LEARNING_RATE = config['learning_rate']
NUM_CLASSES = len(EMOJIS)
NUM_IMAGES_TO_DISPLAY = config['num_images_to_display']
GRID_SIZE = config['grid_size']
RANDOM_SEED = config.get('random_seed', 42)

# Establecer semilla para reproducibilidad
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def render_emoji(emoji, size=IMAGE_SIZE, font_path=FONT_PATH):
    """
    Renderiza un emoji a una imagen en escala de grises de tamaño especificado con variaciones aleatorias.
    """
    # Crear una imagen en escala de grises más grande para acomodar el emoji
    img_size = 64
    img = Image.new('L', (img_size, img_size), color=255)  # 'L' mode para escala de grises
    draw = ImageDraw.Draw(img)

    try:
        # Aleatorizar ligeramente el tamaño de la fuente
        font_size = random.randint(config['font_size_min'], config['font_size_max'])
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        raise IOError(f"No se encontró el archivo de fuente. Por favor verifica FONT_PATH: {font_path}")

    # Calcular la posición para centrar el emoji usando textbbox
    bbox = draw.textbbox((0, 0), emoji, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Introducir desplazamientos aleatorios en la posición
    max_shift = config['max_shift']  # píxeles
    shift_x = random.randint(-max_shift, max_shift)
    shift_y = random.randint(-max_shift, max_shift)

    position = (
        (img_size - text_width) // 2 - bbox[0] + shift_x,
        (img_size - text_height) // 2 - bbox[1] + shift_y
    )

    draw.text(position, emoji, font=font, fill=0)  # Emoji negro sobre fondo blanco

    # Aplicar rotación aleatoria
    rotation_angle = random.uniform(config['rotation_angle_min'], config['rotation_angle_max'])  # grados
    img = img.rotate(rotation_angle, resample=Image.BILINEAR, expand=False, fillcolor=255)

    # Redimensionar al tamaño deseado con remuestreo de alta calidad
    img = img.resize((size, size), Image.LANCZOS)
    # Convertir a binario aplicando un umbral
    threshold = config['threshold']  # Puedes ajustar este valor entre 0 y 1
    img_array = np.array(img).astype(np.float32) / 255.0  # Normalizar a [0,1]
    img_binary = (img_array > threshold).astype(np.float32)  # Conversión binaria
    img_flat = img_binary.flatten()
    return img_flat

def create_emoji_dataset(emojis, samples_per_emoji=SAMPLES_PER_EMOJI):
    """
    Crea un conjunto de datos de emojis renderizados.
    """
    X = []
    y = []
    for idx, emoji in enumerate(tqdm(emojis, desc="Renderizando Emojis")):
        for _ in range(samples_per_emoji):
            img = render_emoji(emoji)
            X.append(img)
            y.append(idx)
    X = np.array(X)
    y = np.array(y)
    return X, y

def display_sample_emojis(emoji, num_samples=5):
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        img_flat = render_emoji(emoji)
        img = img_flat.reshape((IMAGE_SIZE, IMAGE_SIZE))
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.suptitle(f'Variaciones del Emoji: {emoji}')
    plt.show()

def plot_reconstructed_images(vae, X_test, num_images=10):
    reconstructed = []
    for i in range(num_images):
        x = X_test[i].reshape(-1, 1)
        y_rec = vae.reconstruct(x)
        reconstructed.append(y_rec.flatten())
    reconstructed = np.array(reconstructed)

    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))
    for i in range(num_images):
        # Imágenes originales
        axes[0, i].imshow(X_test[i].reshape(IMAGE_SIZE, IMAGE_SIZE), cmap='gray')
        axes[0, i].axis('off')
        # Imágenes reconstruidas
        axes[1, i].imshow(reconstructed[i].reshape(IMAGE_SIZE, IMAGE_SIZE), cmap='gray')
        axes[1, i].axis('off')
    plt.suptitle('Originales (Arriba) vs Reconstruidas (Abajo)')
    plt.tight_layout()
    plt.show()

def generate_images(vae, num_images=10):
    # Muestrear vectores latentes de una distribución normal estándar
    z = np.random.randn(vae.latent_size, num_images)
    generated = []
    for i in range(num_images):
        zi = z[:, i].reshape(-1, 1)
        y_gen = vae.generate(zi)
        generated.append(y_gen.flatten())
    generated = np.array(generated)

    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 2))
    for i in range(num_images):
        axes[i].imshow(generated[i].reshape(IMAGE_SIZE, IMAGE_SIZE), cmap='gray')
        axes[i].axis('off')
    plt.suptitle('Imágenes Generadas')
    plt.tight_layout()
    plt.show()

def plot_loss(vae):
    """
    Grafica la pérdida de entrenamiento a lo largo de las épocas.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(vae.loss_history) + 1), vae.loss_history, marker='o', linestyle='-')
    plt.title('Pérdida de Entrenamiento del VAE a lo largo de las Épocas')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_latent_space(vae, X, y, num_classes=NUM_CLASSES):
    """
    Grafica las representaciones en el espacio latente 2D de los datos.
    """
    # Transponer X para coincidir con la forma de entrada esperada para el codificador VAE
    X_T = X.T  # Forma: (input_size, num_samples)

    # Codificar X para obtener la media de las variables latentes (mu)
    mu, _ = vae.encode(X_T)  # mu forma: (latent_size, num_samples)

    # Transponer mu para obtener forma (num_samples, latent_size)
    latent_mu = mu.T  # Forma: (num_samples, latent_size)

    # Crear un colormap personalizado con solo num_classes colores
    cmap = ListedColormap(plt.cm.tab10.colors[:num_classes])

    # Graficar
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        latent_mu[:, 0],
        latent_mu[:, 1],
        c=y,
        cmap=cmap,
        alpha=0.6,
        s=15
    )
    cbar = plt.colorbar(scatter, ticks=range(num_classes))
    cbar.set_label('Índice de Emoji')
    plt.grid(True)
    plt.xlabel('Dimensión Latente 1')
    plt.ylabel('Dimensión Latente 2')
    plt.title('Espacio Latente 2D del VAE')
    plt.show()

# Mostrar variaciones de cada emoji
for emoji in EMOJIS:
    display_sample_emojis(emoji, num_samples=5)

# Crear el conjunto de datos
X, y = create_emoji_dataset(EMOJIS, samples_per_emoji=SAMPLES_PER_EMOJI)

# Verificar el conjunto de datos
print(f"Conjunto de datos creado con {X.shape[0]} muestras y {X.shape[1]} características por muestra.")
print(f"Distribución de etiquetas: {np.bincount(y)}")

# Mezclar el conjunto de datos
indices = np.arange(X.shape[0])
np.random.seed(RANDOM_SEED)
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# Dividir en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
split_idx = int(0.8 * X.shape[0])
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Conjunto de entrenamiento: {X_train.shape[0]} muestras")
print(f"Conjunto de prueba: {X_test.shape[0]} muestras")

# Inicializar el VAE
vae = VariationalAutoencoder(INPUT_SIZE, HIDDEN_SIZES, LATENT_SIZE)

# Medir el tiempo antes de que comience el entrenamiento
start_time = time.time()

# Entrenar el VAE
vae.train(X_train, batch_size=BATCH_SIZE, epochs=EPOCHS, learning_rate=LEARNING_RATE)

# Medir el tiempo después de que finaliza el entrenamiento
end_time = time.time()

# Calcular y mostrar el tiempo transcurrido
elapsed_time = end_time - start_time
print(f"Entrenamiento completado en {elapsed_time:.2f} segundos.")

# Graficar la pérdida de entrenamiento
plot_loss(vae)

# Graficar imágenes reconstruidas
plot_reconstructed_images(vae, X_test, num_images=NUM_IMAGES_TO_DISPLAY)

# Generar nuevas imágenes
generate_images(vae, num_images=NUM_IMAGES_TO_DISPLAY)

# Graficar el espacio latente 2D
plot_latent_space(vae, X_test, y_test, num_classes=NUM_CLASSES)

# Generar una cuadrícula de imágenes
grid_image = vae.generate_grid(n=GRID_SIZE, digit_size=IMAGE_SIZE)

plt.figure(figsize=(10, 10))
plt.imshow(grid_image, cmap='gray')
plt.axis('off')
plt.title('Emojis Generados Explorando el Espacio Latente 2D')
plt.tight_layout()
plt.show()
