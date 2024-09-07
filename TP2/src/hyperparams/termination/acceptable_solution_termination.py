from typing import List

from TP2.src.hyperparams.termination.abstract_termination import Termination
from TP2.src.model.individual import Individual

class AcceptableSolutionTermination(Termination):
    def __init__(self, target_fitness: float):
        self.target_fitness = target_fitness

    def reset(self):
        pass

    def should_terminate(self, population: List[Individual], generation: int) -> bool:
        best_individual = max(population, key=lambda ind: ind.fitness())
        return best_individual.fitness() >= self.target_fitness