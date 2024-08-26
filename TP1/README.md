# SIA - Trabajo Práctico 1 - 2024

Instituto Tecnológico de Buenos Aires  
Sistemas de Inteligencia Artificial  
Trabajo Práctico 1 - Métodos de Búsqueda

## Requerimentos
- conda enviroment
## Uso
- Primero modificar el config.json a gusto
- Ejecutar el main.py
### uso del config.json
Los siguientes campos son obligatorios:
- "maps": Lista de mapas a correr. Estos consideran como estan guardados dentro de la carpeta input, se deben agregar como strings
- "search_methods": Lista de metodos a probar para obtener soluciones. El archivo all_configs, provee todas las combinaciones de metodos y heuristicas disponibles.
  
Los siguientes campos son opcionales:
- "runs_per_method": Numero entero que representa la cantidad de veces que va a realizar la busqueda por metodo seleccionado. Luego estos se promedian. En caso de no esta defaultea a 1 run.
- "generate_gif": Este booleano determina si debe o no generar una animacion en formato _.gif_ de la solucion encontrada. En caso de no estar defaultea a **false**.

## Contenidos
Este repositorio contiene la implementación de un motor de búsqueda de soluciones para el juego **Sokoban**. 

## Estructura del Proyecto

- **/src**: Código fuente del proyecto.
- **/docs**: Documentación y presentación.

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
