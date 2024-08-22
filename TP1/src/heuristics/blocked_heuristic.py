import sys
from typing import Callable

from docutils.nodes import target

from TP1.src.state import State


def blocked_heuristic(walls: set, targets: set, other_heuristic: Callable[[State], float]) -> Callable[[State], float]:
    def f(inner_state: State) -> float:
        return other_heuristic(inner_state) if not inner_state.is_blocked(walls, targets) else sys.float_info.max

    return f
