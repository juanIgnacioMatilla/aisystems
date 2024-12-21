from typing import TypedDict
from genetic_algorithms.src.hyperparams.crossover.abstract_crossover import Crossover
from genetic_algorithms.src.hyperparams.mutation.abstract_mutation import Mutation
from genetic_algorithms.src.hyperparams.replacement.abstract_replacement import Replacement
from genetic_algorithms.src.hyperparams.selection.abstract_selection import Selection
from genetic_algorithms.src.hyperparams.termination.abstract_termination import Termination


class Hyperparams(TypedDict):
    selection_strategy: Selection
    crossover_strategy: Crossover
    mutation_strategy: Mutation
    termination_strategy: Termination
    replacement_strategy: Replacement
