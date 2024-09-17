# SIA - Trabajo Práctico 2 - 2024

Instituto Tecnológico de Buenos Aires  
Sistemas de Inteligencia Artificial  
Trabajo Práctico 2 - Algoritmos geneticos

--- 
## Tabla de Contenidos

1. [Requerimentos](#Requerimentos)
2. [Uso](#uso)
    1. [Modificación del `config.json`](#uso-del-configjson)
3. [Contenidos](#contenidos)
4. [Estructura del Proyecto](#estructura-del-proyecto)


----

## Requerimentos
- [conda enviroment](https://www.anaconda.com/download/success)
## Uso
- Primero modificar el config.json a gusto
- Ejecutar el main.py

Se pueden ajustar los parametros e hiperparametros utilizando el config.json

Los parametros a modificar:

"runs": cantidad de runs para correr,
"individual_type": tipo de personaje,
"total_points": cantidad de puntos a distribuir,
"population_size": tamaño de poblacion,
"time_limit": tiempo limite para que corra (entre 60 y 120s),

Los hiperparametros a modificar:
opciones para seleccion: 
    "elite",
    "roulette",
    "universal",
    "ranking",
    "boltzmann",
    "deterministic_tournament": DeterministicTournamentSelection,
    "probabilistic_tournament": ProbabilisticTournamentSelection,
    "combined": CombinedSelection
Ejemplo:
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



## Contenidos
Este repositorio contiene la implementación de un motor de algoritmos geneticos para encontrar buenas configuraciones para la creacion de personajes en el juego **ITBUM ONLINE**. 


## Estructura del Proyecto

- `/src`: Código fuente del proyecto.
- `/docs`: Documentación y presentación.
- `/utils`: Parseo del config json.

