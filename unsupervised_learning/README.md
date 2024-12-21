# TP4
## Kohonen:
Este se puede visualizar los agrupamientos y el UDM corriendo main_kohonen.py:
## PCA:
Este se puede visualizar lo pedido corriendo main_pca.py
## OJA:
Este se puede visualizar lo pedido corriendo main_oja.py
## Hopfield:
Este se puede correr la red para las matrices 5x5 en run_hopfield/main_hopfield.py
y los parametros se setean en el config.json
```json
{
    "pattern_names": ["letter_A", "letter_B", "letter_C", "letter_D"],
    "input_pattern_name": "letter_A",
    "max_iters": 100
}
```
Para la parte de PCA con Hopfield hubo diferentes archivos que corren diferentes plots y diferentes outputs de la presentacion.
Estan todos en la carpeta tests_hopfield
## RBM 
Este se puede correr entrando a run_rbm/main_rbm.py y ajustando los parametros del config.json.
visualization es la cantidad de patterns que se quieren ouputear. 
```json
{
    "model": {
        "n_hidden": 64
    },
    "training": {
        "epochs": 5,
        "batch_size": 10,
        "learning_rate": 0.01,
        "k": 1
    },
    "noise": {
        "noise_level": 0.07
    },
    "visualization": {
        "num_images": 15
    }
}
```
Para la parte de RBM hubo diferentes archivos que corren diferentes plots y diferentes outputs de la presentacion.
Estan todos en la carpeta src/model/botlzman
## DBN
Este se puede correr entrando a run_dbn/main_dbn.py y ajustando los parametros del config.json.
visualization es la cantidad de patterns que se quieren ouputear.

La arquitectura de la red se da por layer_sizes donde cada numero del array es la cantidad de neuronas por capa.
 Donde en este ejemplo el 64 es la cantidad de neuronas de la capa hidden de la primera RBM, y la capa visible de la segunda RBM 
```json
{
    "model": {
        "layer_sizes": [784, 64, 10, 10]
    },
    "training": {
        "epochs": 1,
        "batch_size": 10,
        "learning_rate": 0.01,
        "k": 1
    },
    "noise": {
        "noise_level": 0.07
    },
    "visualization": {
        "num_images": 15
    }
}
```
Para la parte de DBN hubo diferentes archivos que corren diferentes plots y diferentes outputs de la presentacion.
Estan todos en la carpeta src/model/botlzman
