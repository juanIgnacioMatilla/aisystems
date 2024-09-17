from typing import List

from TP2.src.hyperparams.termination.abstract_termination import Termination
from TP2.src.model.individual import Individual

class ContentStabilityTermination(Termination):

    def __init__(self, no_improvement_generations: int):
        self.no_improvement_generations = no_improvement_generations
        self.best_fitness = None
        self.stable_generations = 0
        self.reset()  # Initialize the internal state

    def reset(self):
        """Reset the internal state of the termination strategy."""
        self.best_fitness = None
        self.stable_generations = 0


    def should_terminate(self, population: List[Individual], generation: int) -> bool:
        best_individual = max(population, key=lambda ind: ind.fitness())
        current_best_fitness = best_individual.fitness()

        if self.best_fitness is None or current_best_fitness > self.best_fitness:
            self.best_fitness = current_best_fitness
            self.stable_generations = 0
        else:
            self.stable_generations += 1
            if self.stable_generations % 50 == 0:
                print(self.stable_generations)
        return self.stable_generations >= self.no_improvement_generations