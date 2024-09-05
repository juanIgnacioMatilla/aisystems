from TP2.src.hyperparams.crossover.one_point_crossover import OnePointCrossover
from TP2.src.hyperparams.mutation.single_gene_mutation import SingleGeneMutation
from TP2.src.hyperparams.replacement.fill_all_replacement import FillAllReplacement
from TP2.src.hyperparams.selection.elite_selection import EliteSelection
from TP2.src.hyperparams.termination.gen_amount_termination import GenAmountTermination

SELECTION_MAP = {
    "elite": EliteSelection,
}

CROSSOVER_MAP = {
    "one_point": OnePointCrossover,
}

REPLACEMENT_MAP = {
    "fill_all": FillAllReplacement,
}

MUTATION_MAP = {
    "single_gene": SingleGeneMutation,
}

TERMINATION_MAP = {
    "generation_amount": GenAmountTermination,
}
