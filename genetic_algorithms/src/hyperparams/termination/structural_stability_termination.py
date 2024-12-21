from typing import List

from genetic_algorithms.src.hyperparams.termination.abstract_termination import Termination
from genetic_algorithms.src.model.individual import Individual

class StructuralStabilityTermination(Termination):
    def __init__(self, threshold: float, stability_generations: int):
        self.threshold = threshold  # Proportion of population that must remain unchanged
        self.stability_generations = stability_generations
        self.stable_generations = 0
        self.last_population_snapshot = None

    def reset(self):
        self.stable_generations = 0
        self.last_population_snapshot = None

    def should_terminate(self, population: List[Individual], generation: int) -> bool:
        if self.last_population_snapshot is None:
            self.last_population_snapshot = population.copy()

        unchanged_individuals = sum(
            1 for i, ind in enumerate(population)
            if ind.chromosome == self.last_population_snapshot[i].chromosome
        )

        if unchanged_individuals / len(population) >= self.threshold:
            self.stable_generations += 1
        else:
            self.stable_generations = 0

        self.last_population_snapshot = population.copy()

        return self.stable_generations >= self.stability_generations
