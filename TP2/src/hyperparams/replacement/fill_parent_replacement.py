from typing import List

from TP2.src.hyperparams.replacement.abstract_replacement import Replacement
from TP2.src.model.individual import Individual


class FillParentReplacement(Replacement):
    def replace(self, population: List[Individual], offspring: List[Individual]) -> List[Individual]:
        if len(offspring) >= len(population):
            return sorted(offspring, key=lambda ind: ind.fitness(), reverse=True)[:len(population)]
        else:
            remaining = len(population) - len(offspring)
            best_of_population = sorted(population, key=lambda ind: ind.fitness(), reverse=True)[:remaining]
            return offspring + best_of_population
