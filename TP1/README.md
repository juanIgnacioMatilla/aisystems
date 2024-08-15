# SIA - Trabajo Práctico 1 - 2024

Instituto Tecnológico de Buenos Aires  
Sistemas de Inteligencia Artificial  
Trabajo Práctico 1 - Métodos de Búsqueda

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

- **BFS (Breadth-First Search)**
- **DFS (Depth-First Search)**
- **Greedy Search**
-  **A-star Search**
- **IDDFS (Iterative Deepening Depth-First Search) - Opcional**

## Heurísticas

Se implementaron al menos dos heurísticas admisibles y se realizaron pruebas opcionales con heurísticas no admisibles.

## Resultados

El motor de búsqueda proporciona los siguientes resultados al finalizar el procesamiento:

- **Éxito/Fracaso** (si es aplicable)
- **Costo de la solución**
- **Cantidad de nodos expandidos**
- **Cantidad de nodos en la frontera**
- **Solución** (camino desde el estado inicial al estado final)
- **Tiempo de procesamiento**
