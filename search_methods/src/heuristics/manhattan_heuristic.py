from typing import Callable

from search_methods.src.sokoban import Sokoban
from search_methods.src.state import State


def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def manhattan_heuristic(targets: set) -> Callable[[State], float]:
    def f(state: State) -> float:
        sum = 0
        for box in state.boxes:
            distances = [manhattan_distance(box, target) for target in targets]
            sum += min(distances)
        return sum

    return f
