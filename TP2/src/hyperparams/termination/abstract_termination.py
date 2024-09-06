from abc import ABC, abstractmethod
from typing import List

from TP2.src.model.individual import Individual


class Termination(ABC):
    @abstractmethod
    def should_terminate(self, population: List[Individual], generation: int) -> bool:
        pass

    def reset(self):
        """Reset the internal state of the termination strategy."""
        pass