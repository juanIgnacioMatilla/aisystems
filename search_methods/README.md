# SIA - Trabajo Práctico 1 - 2024

Instituto Tecnológico de Buenos Aires  
Sistemas de Inteligencia Artificial  
Trabajo Práctico 1 - Métodos de Búsqueda

--- 
## Tabla de Contenidos

1. [Requerimentos](#Requerimentos)
2. [Uso](#uso)
    1. [Modificación del `config.json`](#uso-del-configjson)
3. [Contenidos](#contenidos)
4. [Estructura del Proyecto](#estructura-del-proyecto)
5. [Sokoban](#juego-implementado)
6. [Métodos de Búsqueda Implementados](#métodos-de-búsqueda-implementados)
7. [Heurísticas](#Heuristicas)
8. [Resultados](#resultados)

----

## Requerimentos
- [conda enviroment](https://www.anaconda.com/download/success)
## Uso
- Primero modificar el config.json a gusto
- Ejecutar el main.py
### uso del config.json
Los siguientes campos son obligatorios:
- **"maps"**: Lista de mapas a correr. Estos consideran como estan guardados dentro de la carpeta input, se deben agregar como strings
- **"search_methods"**: Lista de metodos a probar para obtener soluciones. El archivo all_configs, provee varias combinaciones de metodos y heuristicas disponibles.

Ejemplo:
```json
{
  "maps": ["easy/soko01"],
  "search_methods": [
    {
      "method": "A*",
      "heuristic": "manhattan"
    }
  ]
}
```
  
Los siguientes campos son opcionales:
- **"runs_per_method"**: Numero entero que representa la cantidad de veces que va a realizar la busqueda por metodo seleccionado. Luego estos se promedian. En caso de no esta defaultea a 1 run.
- **"generate_gif"**: Este booleano determina si debe o no generar una animacion en formato _.gif_ de la solucion encontrada. En caso de no estar defaultea a **false**.

```json
{
  "maps": ["easy/soko01"],
  "search_methods": [
    {
      "method": "A*",
      "heuristic": "manhattan"
    }
  ],
  "runs_per_method": 5,
  "generate_gif": true
}
```

Para la heuristica blocked hay que especificar con que heuristica se desea combinar con el parametro "secondary_heuristic". Ejemplo:
```json
 {
       "method": "A*",
       "heuristic": "blocked",
       "secondary_heuristic": "weighted_manhattan"
 }
```

Para la heuristica combined hay que especificar que heuristicas se desea combinar con "combined1", "combined2" y "weight" para indicar el peso de la combinacion. Ejemplo:
```json
 {
   "method": "GGS",
   "heuristic": "combined",
   "combined1": "trivial",
   "combined2": "manhattan",
   "weight": 0.4
 }
```
En este caso la heuristica resultante seria:
$h(e)=0.4*trivial(e)+(1-0.4)*manhattan(e)$

Si quieren usar blocked y de secondary heuristic combined esta seria una posible combinacion:

```json
 {
   "method": "GGS",
   "heuristic": "blocked",
   "secondary_heuristic": "combined",
   "combined1": "trivial",
   "combined2": "manhattan",
   "weight": 0.4 
 }
```

## Contenidos
Este repositorio contiene la implementación de un motor de búsqueda de soluciones para el juego **Sokoban**. 

## Estructura del Proyecto

- `/src`: Código fuente del proyecto.
- `/docs`: Documentación y presentación.

## Juego Implementado

### Sokoban
Sokoban es un juego de rompecabezas donde el objetivo es mover las cajas hasta las posiciones objetivo en un tablero. 
Este motor de búsqueda optimiza la cantidad de movimientos necesarios para resolver el tablero.

## Métodos de Búsqueda Implementados

- **BFS (Breadth-First Search)** -> BFS
- **DFS (Depth-First Search)** -> DFS
- **Greedy Search** -> GGS
-  **A-star Search** -> A*
 
## Heuristicas
- trivial
- blocked
- manhattan
- weighted_manhattan
- minimum_matching
- box_in_target
- combined

## Resultados

El motor de búsqueda proporciona los siguientes resultados al finalizar el procesamiento:

- **Éxito/Fracaso** (si es aplicable)
- **Costo de la solución**
- **Cantidad de nodos expandidos**
- **Cantidad de nodos en la frontera**
- **Solución** (acciones tomadas desde el estado inicial al estado final)
- **Tiempo de procesamiento**
