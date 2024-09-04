from abc import ABC, abstractmethod
from typing import List

from TP2.model.individual import Individual


class Selection(ABC):
    @abstractmethod
    def select(self, population: List[Individual]) -> List[Individual]:
        pass
