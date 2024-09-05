from abc import ABC, abstractmethod

from TP2.src.model.individual import Individual


class Mutation(ABC):
    def __init__(self, p_m: float):
        """
        :param p_m: probability of mutating a gene
        """
        self.p_m = p_m
    @abstractmethod
    def mutate(self, ind1: Individual) -> Individual:
        pass
