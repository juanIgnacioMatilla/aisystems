from typing import Callable

from TP1.src.heuristics.manhattan_heuristic import manhattan_heuristic
from TP1.src.sokoban import Sokoban
from TP1.src.state import State

def weighted_manhattan_heuristic(state: State, targets: set, walls: set) -> Callable[[State], float]:
    def f(inner_state: State = state) -> float:
        penalty = 0
        for bx, by in inner_state.boxes:
            if (bx - 1, by) in walls or (bx + 1, by) in walls:
                penalty += 2  # Penalty for being adjacent to a vertical wall
            if (bx, by - 1) in walls or (bx, by + 1) in walls:
                penalty += 2  # Penalty for being adjacent to a horizontal wall
        return manhattan_heuristic(inner_state, targets)(inner_state) + penalty

    return f
