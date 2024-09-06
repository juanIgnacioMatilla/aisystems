from abc import ABC, abstractmethod
from typing import Tuple

from TP2.src.model.individual import Individual


class Crossover(ABC):
    @abstractmethod
    def crossover(self, ind1: Individual, ind2: Individual) -> Tuple[Individual, Individual]:
        pass

    def reset(self):
        """Reset the internal state of the crossover strategy."""
        pass