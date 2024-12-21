from typing import Callable

from search_methods.src.state import State


def box_in_target_heuristic(targets: set) -> Callable[[State], float]:
    def f(state: State) -> float:
        counter = len(state.boxes)
        for box in state.boxes:
            if box in targets:
                counter -= 1
        return counter

    return f