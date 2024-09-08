from typing import List

from TP2.src.hyperparams.replacement.abstract_replacement import Replacement
from TP2.src.model.individual import Individual


class FillAllReplacement(Replacement):
    def replace(self, population: List[Individual], offspring: List[Individual]) -> List[Individual]:
        combined_population = population + offspring
        # Utiliza el método de selección para elegir individuos de la población combinada
        new_population = self.selection_strategy.select(combined_population)
        return new_population
