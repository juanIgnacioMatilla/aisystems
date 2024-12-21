from typing import Callable

from search_methods.src.heuristics.blocked_heuristic import blocked_heuristic
from search_methods.src.heuristics.box_in_target_heuristic import box_in_target_heuristic
from search_methods.src.heuristics.combined_heuristic import combined_heuristic
from search_methods.src.heuristics.manhattan_heuristic import manhattan_heuristic
from search_methods.src.heuristics.minimum_matching_distance_heuristic import minimum_matching_distance_heuristic
from search_methods.src.heuristics.pseudo_deadlock_heuristic import pseudo_deadlock_heuristic
from search_methods.src.heuristics.trivial_heuristic import trivial_heuristic
from search_methods.src.heuristics.weighted_manhattan_heuristic import weighted_manhattan_heuristic
from search_methods.src.sokoban import Sokoban
from search_methods.src.state import State


class HeuristicBuilder:
    def __init__(self, game: Sokoban):
        self.game = game
        self.heuristic_dict = {"manhattan": self.get_manhattan, "blocked": self.get_blocked,
                               "trivial": self.get_trivial, "box_in_target": self.get_box_in_target_heuristic,
                               #"pseudo_deadlock": self.get_pseudo_deadlock_heuristic,
                               "minimum_matching": self.get_minimum_matching_distance_heuristic,
                               "weighted_manhattan": self.get_weighted_manhattan,
                               "combined": self.get_combined_heuristic}
                               # "blocked_precalculated": self.get_blocked_precalculated}

    @staticmethod
    def get_trivial(weight: float, heuristic1: Callable[[State], float] | None, heuristic2: Callable[[State], float] | None) -> Callable[[State], float]:
        return trivial_heuristic

    def get_blocked(self, weight: float, heuristic1: Callable[[State], float] | None, heuristic2: Callable[[State], float] | None) -> Callable[[State], float]:
        return blocked_heuristic(self.game.walls, self.game.targets, heuristic1)

    def get_manhattan(self, weight: float, heuristic1: Callable[[State], float] | None, heuristic2: Callable[[State], float] | None) -> Callable[[State], float]:
        return manhattan_heuristic(self.game.targets)

    def get_box_in_target_heuristic(self, weight: float, heuristic1: Callable[[State], float] | None, heuristic2: Callable[[State], float] | None) -> Callable[[State], float]:
        return box_in_target_heuristic(self.game.targets)

    def get_minimum_matching_distance_heuristic(self, weight: float, heuristic1: Callable[[State], float] | None, heuristic2: Callable[[State], float] | None) -> Callable[[State], float]:
        return minimum_matching_distance_heuristic(self.game.targets)

    def get_weighted_manhattan(self, weight: float, heuristic1: Callable[[State], float] | None, heuristic2: Callable[[State], float] | None) -> Callable[[State], float]:
        return weighted_manhattan_heuristic(self.game.targets, self.game.walls)

    def get_combined_heuristic(self, weight: float, heuristic1: Callable[[State], float] | None, heuristic2: Callable[[State], float] | None) -> Callable[[State], float]:
        h1 = heuristic1
        h2 = heuristic2
        return combined_heuristic(weight, h1, h2)

    # def get_blocked_precalculated(self, weight: float, heuristic1: Callable[[State], float] | None, heuristic2: Callable[[State], float] | None) -> Callable[[State], float]:
    #     return blocked_precalculated_heuristic(self.game.walls, self.game.targets, self.game.boxes,
    #                                            self.game.free_adjacent_to_wall,
    #                                            other_heuristic)

    def get_heuristic(self, heuristic: str, secondary_heuristic: str, weight: float, combination1: str, combination2: str) -> Callable[[State], float] | None:
        if heuristic is None:
            return None
        if secondary_heuristic is None and weight is None:
            return self.heuristic_dict[heuristic](None, None, None)
        if secondary_heuristic is None and weight is not None:
            return self.heuristic_dict[heuristic](weight, self.heuristic_dict[combination1](None, None, None), self.heuristic_dict[combination2](None, None, None))
        if secondary_heuristic is not None and weight is None:
            return self.heuristic_dict[heuristic](None, self.heuristic_dict[secondary_heuristic](None, None, None), None)
        if weight is not None and combination1 is not None and combination2 is not None:
            return self.heuristic_dict[heuristic](weight, self.heuristic_dict[secondary_heuristic](weight, self.heuristic_dict[combination1](None, None, None), self.heuristic_dict[combination2](None, None, None)), None)
        else:
            return self.heuristic_dict[heuristic](weight, self.heuristic_dict[secondary_heuristic], None)



    def get_pseudo_deadlock_heuristic(self, weight: float, heuristic1: Callable[[State], float] | None, heuristic2: Callable[[State], float] | None) -> Callable[[State], float]:
        return pseudo_deadlock_heuristic(self.game.targets, self.game.walls)