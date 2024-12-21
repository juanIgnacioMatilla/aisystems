from abc import ABC, abstractmethod
from typing import List
from genetic_algorithms.src.hyperparams.selection.abstract_selection import Selection
from genetic_algorithms.src.model.individual import Individual


class Replacement(ABC):
    def __init__(self, selection_strategy: Selection):
        """
        :param selection_strategy: The selection strategy to use for replacement when needed.
        """
        self.selection_strategy = selection_strategy

    @abstractmethod
    def replace(self, population: List[Individual], offspring: List[Individual]) -> List[Individual]:
        pass

    def reset(self):
        """Reset the internal state of the replacement strategy, including the selection strategy."""
        self.selection_strategy.reset()
