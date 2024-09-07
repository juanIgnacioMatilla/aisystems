import random
from typing import List

import numpy as np

from TP2.src.hyperparams.selection.abstract_selection import Selection
from TP2.src.model.individual import Individual


class UniversalSelection(Selection):
    def select(self, population: List[Individual]) -> List[Individual]:
        total_fitness = sum(ind.fitness() for ind in population)
        relative_fitness = [ind.fitness() / total_fitness for ind in population]
        cumulative_fitness = np.cumsum(relative_fitness)

        r = random.random()  # U[0,1)
        selected = []
        for j in range(self.k):
            r_j = r + j / self.k
            for i, cum_fitness in enumerate(cumulative_fitness):
                if r_j < cum_fitness:
                    selected.append(population[i])
                    break
        return selected
