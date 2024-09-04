from abc import ABC, abstractmethod

from TP2.model.individual import Individual


class Mutation(ABC):
    @abstractmethod
    def mutate(self, ind1: Individual) -> Individual:
        pass
