# TP 3 - Perceptron Simple y Multicapa

## Ejercicio 2:
Este puede correrse con el siguiente config file:
```
{
  "k": [k-fold chunks],
  "p_type": [p_type],
  "learning_rate": [learning_rate_value],
  "epochs": [epochs_value]
}
```
A continuacion explicado los valores que toman:
- k: Define la cantidad de divisiones que habra para el k-fold. Los valores aceptables son $k \in (2,28)$
- learning rate: es un double que representa el ratio de aprendizaje.
- epochs: Cantidad de epochs que se entrenara el modelo
- p_type: Define el perceptron que se utilizara
  - "linear": Perceptron lineal
  - "non_linear": Perceptron no lineal con la siguiente funcion de activacion(sigmoidea):

$$f(x) = \frac{1}{1 + e^{-x}}$$

Caso de uso:
```json
{
  "k": 28,
  "p_type": "non_linear",
  "learning_rate": 0.0001,
  "epochs": 30000
}
```
## Ejercicio 3:

Para los puntos 2 y 3 de este ejercicio podemos correrlos utilizando un archivo de configuracion como el siguiente:

```
[
  {
    "inner_layers": [inner_layers_value],
    "learning_rate": [learning_rate_value],
    "optimizer": [optimizer value],
    "epochs": [epochs_value],
    *"alpha": [alpha_value]*
  }
]
```
Donde podemos en el array colocar varios objetos con los atributos alli demostrados. A continuacion explicado los valores que toman:

- inner layers: es un array de enteros donde cada valor i, es la cantidad de neuronas en la i-esima inner layer
  - default:
    - Para el 3.2 es [10,5]
    - Para el 3.3 es [20]
- learning rate: es un double que representa el ratio de aprendizaje.
  - default: 0.1
- optimizer: Cual es el optimizador que usa el perceptron multicapa.
  - "vanilla"
  - "momentum"
  - "adam"
  - default: "vanilla"
- epochs: Cantidad de epochs que se entrenara el modelo
- alpha: Este valor es opcional ya que corresponde al alpha que es parte del optimizador Momentum.
  - default: 0.9
Caso de uso para el 3.3:

```json
[
  {
    "inner_layers": [20],
    "learning_rate": 0.1,
    "optimizer": "vanilla",
    "epochs": 5000
  }
]
```
