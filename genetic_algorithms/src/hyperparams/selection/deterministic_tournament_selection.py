import random
from typing import List
from genetic_algorithms.src.hyperparams.selection.abstract_selection import Selection
from genetic_algorithms.src.model.individual import Individual


class DeterministicTournamentSelection(Selection):
    def __init__(self, k: int, tournament_size: int):
        """
        :param tournament_size: amount of individuals per tournament
        """
        super().__init__(k)
        self.tournament_size = tournament_size

    def select(self, population: List[Individual]) -> List[Individual]:
        selected = []
        for _ in range(self.k):
            tournament = random.sample(population, self.tournament_size)
            best_individual = max(tournament, key=lambda ind: ind.fitness())
            selected.append(best_individual)
        return selected
