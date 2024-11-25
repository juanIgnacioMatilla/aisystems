# TP5
## Autoencoder:
Este se puede visualizar lo pedido corriendo test/autoencoder/fonth_test.py.

El archivo config.json contiene todos los parámetros configurables para el entrenamiento del autoencoder.

Ejemplo de config.json:
```json
{
    "num_runs": 10,
    "learning_rate": 0.001,
    "num_epochs": 4500,
    "error_bar_epoch_interval": 500,
    "input_size": 35,
    "hidden_layers": [60, 50, 30],
    "latent_size": 2,
    "seed": 42
}
```
### Descripción de los parámetros:
* num_runs: Número de veces que se ejecutará el entrenamiento para promediar resultados.
* learning_rate: Tasa de aprendizaje.
* num_epochs: Número de épocas de entrenamiento.
* error_bar_epoch_interval: Intervalo de épocas para registrar el error máximo de píxel.
* input_size: Tamaño de la entrada (por defecto 35 para caracteres de 7x5 píxeles).
* hidden_layers: Lista con los tamaños de las capas ocultas antes de la capa latente.
* latent_size: Tamaño del espacio latente (dimensión de la capa latente).
* seed: Semilla para la reproducibilidad de los resultados.
## Denoising Autoencoder:
Este se puede visualizar lo pedido corriendo test/denoising_autoencoder/fonth_test.py

El archivo config.json contiene todos los parámetros configurables para el entrenamiento del denoising autoencoder.

Ejemplo de config.json:
```json
{
    "input_size": 35,
    "hidden_layers": [60, 50, 30],
    "latent_size": 2,
    "learning_rate": 0.001,
    "num_epochs": 5000,
    "num_runs": 10,
    "noise_levels": [0.1, 0.3, 0.5],
    "num_examples": 5,
    "seed": 42
}
```
Descripción de los parámetros:
* input_size: Tamaño de la entrada (por ejemplo, 35 para caracteres de 7x5 píxeles).
* hidden_layers: Lista con los tamaños de las capas ocultas antes de la capa latente.
* latent_size: Dimensión del espacio latente.
* learning_rate: Tasa de aprendizaje para el optimizador.
* num_epochs: Número de épocas de entrenamiento.
* num_runs: Número de ejecuciones por cada nivel de ruido para promediar resultados.
* noise_levels: Lista de niveles de ruido a probar (por ejemplo, [0.1, 0.3, 0.5]).
* num_examples: Número de ejemplos a mostrar en los resultados de denoising.
* seed: Semilla para reproducibilidad de los resultados.
## Variational Autoencoder:
Este se puede visualizar lo pedido corriendo test/variational_autoencoder/emoji_test.py

El archivo config.json contiene todos los parámetros configurables para el entrenamiento del variational autoencoder.

```json
{
    "font_path": "C:\\Windows\\Fonts\\seguiemj.ttf",
    "emojis": ["😍", "😎", "⚽", "🙃", "👽"],
    "image_size": 20,
    "samples_per_emoji": 1000,
    "hidden_sizes": [256, 128],
    "latent_size": 2,
    "batch_size": 2,
    "epochs": 20,
    "learning_rate": 0.001,
    "num_images_to_display": 10,
    "grid_size": 15,
    "random_seed": 42,
    "font_size_min": 44,
    "font_size_max": 52,
    "max_shift": 3,
    "rotation_angle_min": -7,
    "rotation_angle_max": 7,
    "threshold": 0.5
}
```

Descripción de los parámetros:

* font_path: Ruta al archivo de fuente que soporta emojis. Para Mac, "/System/Library/Fonts/Apple Color Emoji.ttc"
Para linux, habria que descargar una fuente, como Noto Color Emoji
* emojis: Lista de emojis a incluir en el conjunto de datos.
* image_size: Tamaño de las imágenes (ancho y alto en píxeles).
* samples_per_emoji: Número de muestras a generar por cada emoji.
* hidden_sizes: Lista de tamaños de las capas ocultas en el VAE.
* latent_size: Tamaño del espacio latente (dimensión de la capa latente).
* batch_size: Tamaño del lote para el entrenamiento.
* epochs: Número de épocas de entrenamiento.
* learning_rate: Tasa de aprendizaje para el optimizador.
* num_images_to_display: Número de imágenes a mostrar en las visualizaciones.
* grid_size: Tamaño de la cuadrícula para generar imágenes explorando el espacio latente.
* random_seed: Semilla para reproducibilidad.
* font_size_min y font_size_max: Rango de tamaños de fuente para variación aleatoria.
* max_shift: Desplazamiento máximo en píxeles para variaciones de posición.
* rotation_angle_min y rotation_angle_max: Rango de ángulos de rotación en grados.
* threshold: Umbral para convertir imágenes a binarias.
