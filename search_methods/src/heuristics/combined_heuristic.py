#make a heuristic that combines two heuristics

from typing import Callable

from search_methods.src.state import State


def combined_heuristic(weight: float, heuristic1: Callable[[State], float], heuristic2: Callable[[State], float]) -> Callable[[State], float]:
    if weight < 0 or weight > 1:
        raise ValueError("Weight must be between 0 and 1")

    def f(state: State) -> float:
        return weight * heuristic1(state) + (1 - weight) * heuristic2(state)
    return f