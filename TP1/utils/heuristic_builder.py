import sys
from typing import Callable

from TP1.src.heuristics.blocked_heuristic import blocked_heuristic
from TP1.src.heuristics.box_in_target_heuristic import box_in_target_heuristic
from TP1.src.heuristics.manhattan_heuristic import manhattan_heuristic
from TP1.src.heuristics.trivial_heuristic import trivial_heuristic
from TP1.src.sokoban import Sokoban
from TP1.src.state import State


class HeuristicBuilder:
    def __init__(self, game: Sokoban):
        self.game = game
        self.heuristic_dict = {"manhattan": self.get_manhattan, "blocked": self.get_blocked, "trivial": self.get_trivial, "box_in_target": self.get_box_in_target_heuristic }


    @staticmethod
    def get_trivial(other_heuristic: Callable[[State], float] | None) -> Callable[[State], float]:
        return trivial_heuristic

    def get_blocked(self, other_heuristic:Callable[[State], float]) -> Callable[[State], float]:
        return blocked_heuristic(self.game.walls, other_heuristic)



    def get_manhattan(self, other_heuristic:Callable[[State], float] | None)-> Callable[[State], float]:
        return manhattan_heuristic(self.game.targets)

    def get_box_in_target_heuristic(self, other_heuristic:Callable[[State], float] | None):
            return box_in_target_heuristic(self.game.targets)

    def get_heuristic(self, heuristic: str, other:str)-> Callable[[State], float] | None:
        if heuristic is None:
            return None
        other = self.heuristic_dict[other](None) if other else None
        return self.heuristic_dict[heuristic](other)