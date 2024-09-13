from typing import List

from TP2.src.hyperparams.termination.abstract_termination import Termination
from TP2.src.model.individual import Individual


class GenAmountTermination(Termination):
    def __init__(self, amount: int):
        self.gen_amount = amount

    def should_terminate(self, population: List[Individual], generation: int) -> bool:
        return generation == self.gen_amount
