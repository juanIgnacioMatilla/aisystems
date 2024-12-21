# SIA - Trabajo Práctico 2 - 2024

Instituto Tecnológico de Buenos Aires  
Sistemas de Inteligencia Artificial  
Trabajo Práctico 2 - Algoritmos geneticos

--- 
## Tabla de Contenidos

1. [Requerimentos](#Requerimentos)
2. [Uso](#uso)
    1. [Modificación del `config.json`](#uso-del-configjson)
       2. [Parametros a modificables](#Parametros-modificables)
       3. [Hiperparametros modificables](#Hiperparametros-modificables)
          4. [Opciones para seleccion](#Opciones-para-seleccion)
          5. [Opciones para crossover](#Opciones-para-crossover)
          6. [Opciones para mutation](#Opciones-para-mutation)
          7. [Options for replacement](#Options-for-replacement)
       8. [Ejemplo](#Ejemplo)
3. [Contenidos](#contenidos)
4. [Estructura del Proyecto](#estructura-del-proyecto)


----

## Requerimentos
- [conda enviroment](https://www.anaconda.com/download/success)
## Uso
- Primero modificar el config.json a gusto
- Ejecutar el main.py

### uso del config.json
Se pueden ajustar los parametros e hiperparametros utilizando el config.json

#### Parametros modificables:

```json
    "runs": cantidad de runs para correr,
    "individual_type": tipo de personaje,
    "total_points": cantidad de puntos a distribuir,
    "population_size": tamaño de poblacion,
    "time_limit": tiempo limite para que corra,
```

> [!NOTE]  
> El valor del tiempo tiene que ser dentro del intervalo [1, 120].

#### Hiperparametros modificables:

  * ##### Opciones para seleccion:
    * `elite`: 
        ```json
        "selection": {
          "name": "elite",
          "k": 50,
        }
        ```
    * `roulette`: 
      ```json
      "selection": {
        "name": "roulette",
        "k": 50,
      }
      ```
    * `universal`: 
      ```json
      "selection": {
        "name": "universal",
        "k": 50,
      }
      ```
    * `ranking`: 
      ```json
      "selection": {
        "name": "ranking",
        "k": 50,
      }
      ```
    * `boltzmann`: 
      ```json
      "selection": {
        "name": "boltzmann",
        "temperature": 60,
      }
      ```
    * `deterministic_tournament`: 
      ```json
      "selection": {
        "name": "deterministic_tournament",
        "k": 50,
        "torunament_size": 5
      }
      ```
    * `probabilistic_tournament`: 
      ```json
      "selection": {
        "name": "probabilistic_tournament",
        "k": 50,
        "threshold": 0.5
      }
      ```

      > [!NOTE]  
      > El valor que puede tomar Treshold es entre [0.5, 1].

    * `combined`: 
      ```json
      "selection": {
          "name": "combined",
          "k": 50,
          "method_a": {
              "name":"elite"
          },
          "method_b": {
              "name":"boltzmann",
              "temperature": 100
          },
          "percentage_a": 0.4
      }
      ```
  * ##### Opciones para crossover:
    * `one_point`: 
      ```json
      "crossover": {
          "name": "one_point"
      }
      ```
    * `two_point`: 
      ```json
        "crossover": {
            "name": "two_point"
        }
        ```
    * `uniform`:
        ```json
        "crossover": {
            "name": "uniform",
            "p_c": 0.5
        }
        ```
    * `anular`:
        ```json
        "crossover": {
            "name": "anular",
            "p_c": 0.5
        }
        ```
  * ##### Opciones para mutation:
    * `total_gene`: 
      ```json
      "mutation": {
          "name": "total_gene",
          "p_m": 0.01
      }
      ```
    * `single_gene`: 
      ```json
      "mutation": {
          "name": "single_gene",
          "p_m": 0.01
      }
      ```
    * `uniform`: 
      ```json
      "mutation": {
          "name": "uniform",
          "p_m": 0.01
      }
      ```
    * `non_uniform`: 
      ```json
      "mutation": {
          "name": "non_uniform",
          "p_m": 0.01
      }
      ```
    * `multi_gen`: 
      ```json
      "mutation": {
          "name": "multi_gen",
          "p_m": 0.01
      }
      ```
  * ##### Options for replacement:
    * `fill_all`: 
      ```json
      "replacement": {
        "name": "fill_all"
      }
      ```
    * `fill_parent`: 
        ```json
        "replacement": {
          "name": "fill_parent"
        }
        ```
  * ##### Optciones para terminacion:
    * `acceptable_solution`: 
      ```json
      "termination": {
        "name": "acceptable_solution",
        "target_fitness": 0.9
      }
      ```
    * `content_stability`: 
        ```json
        "termination": {
          "name": "content_stability",
          "no_improvement_generations": 100
        }
        ```
    * `generation_amount`: 
        ```json
        "termination": {
          "name": "generation_amount",
          "amount": 100
        }
        ```
    * `structure_stability`: 
        ```json
        "termination": {
          "name": "structure_stability",
          "threshold": 0.2,
          "no_improvement_generations": 100
        }
       ```

##### Ejemplo:
```json
  "hyperparams": {
      "selection": {
          "name": "combined",
          "k": 50,
          "method_a": {
              "name":"elite"
          },
          "method_b": {
              "name":"boltzmann",
              "temperature": 100
          },
          "percentage_a": 0.4
      },
      "crossover": {
          "name": "one_point"
      },
      "mutation": {
          "name": "total_gene",
          "p_m": 0.01
      },
      "replacement": {
          "name": "fill_all",
          "selection": {
              "name": "elite"
          }
      },
      "termination": {
          "name": "content_stability",
          "no_improvement_generations": 100
      }
  }
```
Un ejemplo del formato final del json seria:
```json
[
    {
        "runs": 1,
        "individual_type": "ARCHER",
        "total_points": 200,
        "population_size": 100,
        "time_limit": 120.0,
        "hyperparams": {
            "selection": {
                "name": "combined",
                "k": 50,
                "method_a": {
                    "name":"elite"
                },
                "method_b": {
                    "name":"boltzmann",
                    "temperature": 100
                },
                "percentage_a": 0.4
            },
            "crossover": {
                "name": "one_point"
            },
            "mutation": {
                "name": "total_gene",
                "p_m": 0.01
            },
            "replacement": {
                "name": "fill_all",
                "selection": {
                    "name": "elite"
                }
            },
            "termination": {
                "name": "content_stability",
                "no_improvement_generations": 100
            }
        }
    }
]
```

Tambien se pueden correr varias configs secuencialmente, hay un ejemplo en el archivo example_configs.
## Contenidos
Este repositorio contiene la implementación de un motor de algoritmos geneticos para encontrar buenas configuraciones para la creacion de personajes en el juego **ITBUM ONLINE**. 


## Estructura del Proyecto

- `/src`: Código fuente del proyecto.
- `/docs`: Documentación y presentación.
- `/utils`: Parseo del config json.

