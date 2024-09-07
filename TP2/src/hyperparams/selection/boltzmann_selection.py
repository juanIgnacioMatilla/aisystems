import random
from typing import List

import numpy as np

from TP2.src.hyperparams.selection.abstract_selection import Selection
from TP2.src.model.individual import Individual


class BoltzmannSelection(Selection):
    def __init__(self, k: int, temperature: float):
        """
        :param temperature: high temperatures around 200 and low around 1
        """
        super().__init__(k)
        self.temperature = temperature

    def select(self, population: List[Individual]) -> List[Individual]:
        avg_fitness = np.mean([ind.fitness() for ind in population])
        pseudo_fitness = [np.exp(ind.fitness() / self.temperature) / np.exp(avg_fitness / self.temperature)
                          for ind in population]
        cumulative_fitness = np.cumsum(pseudo_fitness)

        selected = []
        for _ in range(self.k):
            r = random.random()  # U[0,1)
            for i, cum_fitness in enumerate(cumulative_fitness):
                if r < cum_fitness:
                    selected.append(population[i])
                    break
        return selected
