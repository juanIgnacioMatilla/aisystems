# from TP2.src.hyperparams.crossover.anular_crossover import AnularCrossover
from TP2.src.hyperparams.crossover.one_point_crossover import OnePointCrossover
from TP2.src.hyperparams.crossover.two_point_crossover import TwoPointCrossover
from TP2.src.hyperparams.crossover.uniform_crossover import UniformCrossover
from TP2.src.hyperparams.mutation.multi_gene_mutation import MultiGeneMutation
from TP2.src.hyperparams.mutation.single_gene_mutation import SingleGeneMutation
from TP2.src.hyperparams.mutation.total_gene_mutation import TotalGeneMutation
from TP2.src.hyperparams.replacement.fill_all_replacement import FillAllReplacement
from TP2.src.hyperparams.replacement.fill_parent_replacement import FillParentReplacement
from TP2.src.hyperparams.selection.boltzmann_selection import BoltzmannSelection
from TP2.src.hyperparams.selection.combined_selection import CombinedSelection
from TP2.src.hyperparams.selection.deterministic_tournament_selection import DeterministicTournamentSelection
from TP2.src.hyperparams.selection.elite_selection import EliteSelection
from TP2.src.hyperparams.selection.probabilistic_tournament_selection import ProbabilisticTournamentSelection
from TP2.src.hyperparams.selection.ranking_selection import RankingSelection
from TP2.src.hyperparams.selection.roulette_selection import RouletteSelection
from TP2.src.hyperparams.selection.universal_selection import UniversalSelection
from TP2.src.hyperparams.termination.gen_amount_termination import GenAmountTermination
from TP2.src.hyperparams.termination.structural_stability_termination import StructuralStabilityTermination
from TP2.src.hyperparams.termination.content_stability_termination import ContentStabilityTermination
from TP2.src.hyperparams.termination.acceptable_solution_termination import AcceptableSolutionTermination
from TP2.src.hyperparams.mutation.uniform_gene_mutation import UniformGeneMutation

SELECTION_MAP = {
    "elite": EliteSelection,
    "roulette": RouletteSelection,
    "universal": UniversalSelection,
    "ranking": RankingSelection,
    "boltzmann": BoltzmannSelection,
    "deterministic_tournament": DeterministicTournamentSelection,
    "probabilistic_tournament": ProbabilisticTournamentSelection,
    "combined": CombinedSelection
}

CROSSOVER_MAP = {
    "one_point": OnePointCrossover,
    # "two_point": TwoPointCrossover,
    # "uniform": UniformCrossover,
    # "anular": AnularCrossover,
}

REPLACEMENT_MAP = {
    "fill_all": FillAllReplacement,
    "fill_parent": FillParentReplacement,
}

MUTATION_MAP = {
    "single_gene": SingleGeneMutation,
    "multi_gene": MultiGeneMutation,
    "uniform_gene": UniformGeneMutation,
    "total_gene": TotalGeneMutation
}

TERMINATION_MAP = {
    "generation_amount": GenAmountTermination,
    "structural_stability": StructuralStabilityTermination,
    "content_stability": ContentStabilityTermination,
    "acceptable_solution": AcceptableSolutionTermination,
}
