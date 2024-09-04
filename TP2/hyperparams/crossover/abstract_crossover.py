from abc import ABC, abstractmethod

from TP2.model.individual import Individual


class Crossover(ABC):
    @abstractmethod
    def crossover(self, ind1: Individual, ind2: Individual) -> Individual:
        pass
