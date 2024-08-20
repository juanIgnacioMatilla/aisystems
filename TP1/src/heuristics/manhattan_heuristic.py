from typing import Callable

from TP1.src.sokoban import Sokoban
from TP1.src.state import State



def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
# def f(state: State) -> float:
#         heuristic_value = 0
#         for box in state.boxes:
#             min_distance = float('inf')
#             for goal in targets:
#                 distance = manhattan_distance(box, goal)
#                 min_distance = min(min_distance, distance)
#             heuristic_value += min_distance
#         return heuristic_value


def manhattan_heuristic(targets: set) -> Callable[[State], float]:
    def f(state: State) -> float:
        sum = 0
        for box in state.boxes:
            distances = [manhattan_distance(box, target) for target in targets]
            sum += min(distances)
        return sum
    return f