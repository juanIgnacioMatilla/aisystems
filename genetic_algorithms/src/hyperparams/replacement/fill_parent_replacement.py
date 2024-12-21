from typing import List
from genetic_algorithms.src.hyperparams.replacement.abstract_replacement import Replacement
from genetic_algorithms.src.model.individual import Individual


class FillParentReplacement(Replacement):
    def replace(self, population: List[Individual], offspring: List[Individual]) -> List[Individual]:
        if len(offspring) >= len(population):
            return self.selection_strategy.select(offspring)
        else:
            remaining = len(population) - len(offspring)
            best_of_population = self.selection_strategy.select(population)[:remaining]
            return offspring + best_of_population
