import random
from typing import List

from TP2.src.hyperparams.selection.abstract_selection import Selection
from TP2.src.model.individual import Individual


class ProbabilisticTournamentSelection(Selection):
    def __init__(self, k: int, threshold: float):
        """
        :param threshold: probability of choosing the best individual (range between 0.5 and 1)
        """
        super().__init__(k)
        self.threshold = threshold

    def select(self, population: List[Individual]) -> List[Individual]:
        selected = []
        for _ in range(self.k):
            ind1, ind2 = random.sample(population, 2)
            if random.random() < self.threshold:
                selected.append(max(ind1, ind2, key=lambda ind: ind.fitness()))
            else:
                selected.append(min(ind1, ind2, key=lambda ind: ind.fitness()))
        return selected
