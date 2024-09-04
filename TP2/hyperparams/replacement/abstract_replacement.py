from abc import ABC, abstractmethod
from typing import List

from TP2.model.individual import Individual


class Replacement(ABC):
    @abstractmethod
    def replace(self, population: List[Individual], offspring: List[Individual]) -> List[Individual]:
        pass
