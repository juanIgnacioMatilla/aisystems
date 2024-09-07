from random import sample
from typing import List

from TP2.src.hyperparams.replacement.abstract_replacement import Replacement
from TP2.src.model.individual import Individual


class FillAllReplacement(Replacement):

    def replace(self, population: List[Individual], offspring: List[Individual]) -> List[Individual]:
        combined = population + offspring
        combined_sorted = sorted(combined, key=lambda ind: ind.fitness(), reverse=True)
        return combined_sorted[:len(population)]
