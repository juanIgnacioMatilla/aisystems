import sys
from typing import Callable

from TP1.src.state import State


def blocked_heuristic( walls: set, other_heuristic: Callable[[State], float]) -> Callable[[State], float]:
    def f(inner_state: State) -> float:
        return other_heuristic(inner_state) if inner_state.is_blocked(walls) else sys.float_info.max

    return f
