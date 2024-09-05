from abc import ABC, abstractmethod
from typing import List

from TP2.src.model.individual import Individual


class Selection(ABC):
    def __init__(self, k: int):
        """
        :param k: number of parents to choose
        """
        self.k = k

    @abstractmethod
    def select(self, population: List[Individual]) -> List[Individual]:
        """
        Selecciona una lista de individuos padres de la población.

        :param population: Lista de individuos en la población actual.
        :param num_parents: Número de padres a seleccionar.
        :param kwargs: Argumentos con nombre adicionales para la estrategia de selección.
        :return: Lista de individuos seleccionados como padres.
        """
        pass
