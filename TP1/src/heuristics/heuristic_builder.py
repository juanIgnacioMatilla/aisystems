import sys
from typing import Callable

from TP1.src.sokoban import Sokoban
from TP1.src.state import State


class HeuristicBuilder:
    def __init__(self, game: Sokoban):
        self.game = game
        self.available_heuristics = ["blocked", "manhattan", "trivial"]

    @staticmethod
    def get_trivial(other_heuristic: Callable[[State], float] | None) -> Callable[[State], float]:
        def f(state: State) -> float:
            return 1
        return f

    def get_blocked(self, other_heuristic:Callable[[State], float]) -> Callable[[State], float]:
        def f(state: State) -> float:
            return other_heuristic(state) if state.is_blocked(self.game.walls) else sys.float_info.max
        return f

    def get_manhattan(self, other_heuristic:Callable[[State], float] | None)-> Callable[[State], float]:
        def f(state: State) -> float:
            return sum(
                min(abs(bx - tx) + abs(by - ty) for tx, ty in self.game.targets) for bx, by in state.boxes
            )
        return f

    def get_heuristic(self, heuristic: str, other:str)-> Callable[[State], float] | None:
        if heuristic is None:
            return None
        secondary = None
        if other == 'manhattan':
            secondary = self.get_manhattan(None)
        elif other == 'trivial':
            secondary = self.get_trivial(None)

        if heuristic == 'blocked':
            return self.get_blocked(secondary)
        elif heuristic == 'manhattan':
            return self.get_manhattan(secondary)
        else:
            return self.get_trivial(secondary)