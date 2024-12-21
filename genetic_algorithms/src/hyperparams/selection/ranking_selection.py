import random
from typing import List

import numpy as np

from genetic_algorithms.src.hyperparams.selection.abstract_selection import Selection
from genetic_algorithms.src.model.individual import Individual


class RankingSelection(Selection):
    def select(self, population: List[Individual]) -> List[Individual]:
        population_size = len(population)
        ranked_population = sorted(population, key=lambda ind: ind.fitness(), reverse=True)

        rank_based_fitness = [(population_size - rank) / population_size for rank in range(population_size)]
        cumulative_rank_fitness = np.cumsum(rank_based_fitness)

        selected = []
        for _ in range(self.k):
            r = random.random()  # U[0,1)
            for i, cum_fitness in enumerate(cumulative_rank_fitness):
                if r < cum_fitness:
                    selected.append(ranked_population[i])
                    break
        return selected
