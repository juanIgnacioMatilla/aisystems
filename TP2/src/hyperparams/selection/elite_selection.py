from typing import List

from TP2.src.hyperparams.selection.abstract_selection import Selection
from TP2.src.model.individual import Individual


class EliteSelection(Selection):

    def select(self, population: List[Individual]) -> List[Individual]:
        # Ordenar la población por fitness en orden descendente y seleccionar los mejores
        sorted_population = sorted(population, key=lambda ind: ind.fitness(), reverse=True)
        return sorted_population[:self.k]
