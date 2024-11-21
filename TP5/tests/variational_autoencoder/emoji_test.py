import os
import random
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from tqdm import tqdm

from TP5.src.model.variational_autoencoder import VariationalAutoencoder

# Define the path to the emoji-supporting font
# Update this path based on your OS and available fonts
# For Windows:
FONT_PATH = "C:\\Windows\\Fonts\\seguiemj.ttf"
# For macOS:
# FONT_PATH = "/System/Library/Fonts/Apple Color Emoji.ttc"
# For Linux, you might need to install a font that supports emojis, such as Noto Color Emoji

# List of emojis to include in the dataset
EMOJIS = [
    '😍',
    '😎',
    '⚽',  # Soccer Ball
    '🙃',
    '👽',
]

# Global variable for image size
IMAGE_SIZE = 20  # You can adjust this value as needed


def render_emoji(emoji, size=IMAGE_SIZE):
    """
    Render an emoji to a grayscale image of specified size with random variations.
    Args:
        emoji (str): The emoji character to render.
        size (int): The size (width and height) of the image in pixels.
    Returns:
        np.ndarray: Flattened grayscale image array of shape (size*size,).
    """
    # Create a grayscale image larger than desired to accommodate the emoji
    img_size = 64
    img = Image.new('L', (img_size, img_size), color=255)  # 'L' mode for grayscale
    draw = ImageDraw.Draw(img)

    try:
        # Randomize font size slightly
        font_size = random.randint(44, 52)
        font = ImageFont.truetype(FONT_PATH, font_size)
    except IOError:
        raise IOError(f"Font file not found. Please check the FONT_PATH: {FONT_PATH}")

    # Calculate the position to center the emoji using textbbox
    bbox = draw.textbbox((0, 0), emoji, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Introduce random shifts in position
    max_shift = 3  # pixels
    shift_x = random.randint(-max_shift, max_shift)
    shift_y = random.randint(-max_shift, max_shift)

    position = (
        (img_size - text_width) // 2 - bbox[0] + shift_x,
        (img_size - text_height) // 2 - bbox[1] + shift_y
    )

    draw.text(position, emoji, font=font, fill=0)  # Black emoji on white background

    # Apply random rotation
    rotation_angle = random.uniform(-7, 7)  # degrees
    img = img.rotate(rotation_angle, resample=Image.BILINEAR, expand=False, fillcolor=255)

    # Resize to desired size with high-quality resampling
    img = img.resize((size, size), Image.LANCZOS)
    # Convert to binary by applying a threshold
    threshold = 0.5  # You can adjust this value between 0 and 1
    img_array = np.array(img).astype(np.float32) / 255.0  # Normalize to [0,1]
    img_binary = (img_array > threshold).astype(np.float32)  # Binary conversion
    img_flat = img_binary.flatten()
    return img_flat


def create_emoji_dataset(emojis, samples_per_emoji=1000):
    """
    Create a dataset of rendered emojis.
    Args:
        emojis (list): List of emoji characters.
        samples_per_emoji (int): Number of samples per emoji.
    Returns:
        X (np.ndarray): Array of shape (num_samples, IMAGE_SIZE*IMAGE_SIZE) with grayscale images.
        y (np.ndarray): Array of labels corresponding to emojis.
    """
    X = []
    y = []
    for idx, emoji in enumerate(tqdm(emojis, desc="Rendering Emojis")):
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
    plt.suptitle(f'Variations of Emoji: {emoji}')
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
        # Original images
        axes[0, i].imshow(X_test[i].reshape(IMAGE_SIZE, IMAGE_SIZE), cmap='gray')
        axes[0, i].axis('off')
        # Reconstructed images
        axes[1, i].imshow(reconstructed[i].reshape(IMAGE_SIZE, IMAGE_SIZE), cmap='gray')
        axes[1, i].axis('off')
    plt.suptitle('Original (Top) vs Reconstructed (Bottom)')
    plt.tight_layout()
    plt.show()


def generate_images(vae, num_images=10):
    # Sample latent vectors from standard normal distribution
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
    plt.suptitle('Generated Images')
    plt.tight_layout()
    plt.show()


def plot_loss(vae):
    """
    Plot the training loss over epochs.

    Args:
        vae (VariationalAutoencoder): Trained VAE model.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(vae.loss_history) + 1), vae.loss_history, marker='o', linestyle='-')
    plt.title('VAE Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_latent_space(vae, X, y, num_classes=5):
    """
    Plot the 2D latent space representations of the data.

    Args:
        vae (VariationalAutoencoder): The trained VAE model.
        X (np.ndarray): Data to encode and plot (shape: [num_samples, input_size]).
        y (np.ndarray): Labels corresponding to the data (shape: [num_samples,]).
        num_classes (int): Number of unique classes (emojis).
    """
    # Transpose X to match expected input shape for the VAE encoder
    X_T = X.T  # Shape: (input_size, num_samples)

    # Encode X to get the mean of the latent variables (mu)
    mu, _ = vae.encode(X_T)  # mu shape: (latent_size, num_samples)

    # Transpose mu to get shape (num_samples, latent_size)
    latent_mu = mu.T  # Shape: (num_samples, latent_size)

    # Create a custom colormap with only 5 colors
    cmap = ListedColormap(plt.cm.tab10.colors[:num_classes])

    # Plotting
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
    cbar.set_label('Emoji Index')
    plt.grid(True)
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('2D Latent Space of VAE')
    plt.show()

for i in range(len(EMOJIS)):
    # Create and display sample variations for an emoji
    display_sample_emojis(EMOJIS[i], num_samples=5)  # Example: First emoji

# Create the dataset
X, y = create_emoji_dataset(EMOJIS, samples_per_emoji=1000)

# Verify dataset
print(f"Dataset created with {X.shape[0]} samples and {X.shape[1]} features per sample.")
print(f"Label distribution: {np.bincount(y)}")

# Shuffle the dataset
indices = np.arange(X.shape[0])
np.random.seed(42)
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# Split into training and testing sets (80% train, 20% test)
split_idx = int(0.8 * X.shape[0])
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# Initialize the VAE
input_size = IMAGE_SIZE ** 2  # e.g., 20x20 pixels
hidden_sizes = [256, 128]  # Increased hidden sizes for better capacity
latent_size = 2  # For 2D latent space visualization

vae = VariationalAutoencoder(input_size, hidden_sizes, latent_size)
# Measure the time before training starts
start_time = time.time()

# Train the VAE
vae.train(X_train, batch_size=2, epochs=20, learning_rate=0.001)  # Adjust epochs or batch size as needed

# Measure the time after training ends
end_time = time.time()

# Calculate and display the elapsed time
elapsed_time = end_time - start_time
print(f"Training completed in {elapsed_time:.2f} seconds.")
# Plot the training loss
plot_loss(vae)

# Plot reconstructed images
plot_reconstructed_images(vae, X_test, num_images=10)

# Generate new images
generate_images(vae, num_images=10)

# Plot the 2D latent space
plot_latent_space(vae, X_test, y_test, num_classes=len(EMOJIS))

# Corrected digit_size from 10 to IMAGE_SIZE
grid_image = vae.generate_grid(n=15, digit_size=IMAGE_SIZE)

plt.figure(figsize=(10, 10))
plt.imshow(grid_image, cmap='gray')
plt.axis('off')
plt.title('Generated Emojis Traversing the 2D Latent Space')
plt.tight_layout()
plt.show()
