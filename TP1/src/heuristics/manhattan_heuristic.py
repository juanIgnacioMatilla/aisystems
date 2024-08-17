from typing import Callable

from TP1.src.sokoban import Sokoban
from TP1.src.state import State


def manhattan_heuristic(state: State, targets: set) -> Callable[[State], float]:
    def f(state: State) -> float:
        return sum(
            min(abs(bx - tx) + abs(by - ty) for tx, ty in targets) for bx, by in state.boxes
        )

    return f
