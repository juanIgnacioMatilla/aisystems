from random import sample
from typing import List

from TP2.src.hyperparams.replacement.abstract_replacement import Replacement
from TP2.src.model.individual import Individual


class FillAllReplacement(Replacement):

    def replace(self, population: List[Individual], offspring: List[Individual]) -> List[Individual]:
        # Combine current population with offspring
        combined_population = population + offspring

        # Select N individuals from the combined population
        next_generation = sample(combined_population, len(population))

        return next_generation
