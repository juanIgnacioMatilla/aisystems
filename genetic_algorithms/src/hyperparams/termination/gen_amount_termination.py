from typing import List

from genetic_algorithms.src.hyperparams.termination.abstract_termination import Termination
from genetic_algorithms.src.model.individual import Individual


class GenAmountTermination(Termination):
    def __init__(self, amount: int):
        self.gen_amount = amount

    def should_terminate(self, population: List[Individual], generation: int) -> bool:
        return generation == self.gen_amount
