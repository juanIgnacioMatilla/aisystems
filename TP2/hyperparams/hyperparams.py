from typing import TypedDict
from TP2.hyperparams.crossover.abstract_crossover import Crossover
from TP2.hyperparams.mutation.abstract_mutation import Mutation
from TP2.hyperparams.replacement.abstract_replacement import Replacement
from TP2.hyperparams.selection.abstract_selection import Selection
from TP2.hyperparams.termination.abstract_termination import Termination


class Hyperparams(TypedDict):
    selection_strategy: Selection
    crossover_strategy: Crossover
    mutation_strategy: Mutation
    termination_strategy: Termination
    replacement_strategy: Replacement
