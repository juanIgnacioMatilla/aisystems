from abc import ABC, abstractmethod
from typing import List

from TP2.src.model.individual import Individual


class Replacement(ABC):
    @abstractmethod
    def replace(self, population: List[Individual], offspring: List[Individual]) -> List[Individual]:
        pass

    def reset(self):
        """Reset the internal state of the replacement strategy."""
        pass