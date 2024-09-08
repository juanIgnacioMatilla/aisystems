from typing import List

from TP2.src.hyperparams.termination.abstract_termination import Termination
from TP2.src.model.individual import Individual


class CombinedTermination(Termination):
    def __init__(self, termination_criteria: List[Termination]):
        self.termination_criteria = termination_criteria

    def should_terminate(self, population: List[Individual], generation: int) -> bool:
        return any(criterion.should_terminate(population, generation) for criterion in self.termination_criteria)
