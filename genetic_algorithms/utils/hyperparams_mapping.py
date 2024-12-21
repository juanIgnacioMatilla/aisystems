from genetic_algorithms.src.hyperparams.crossover.anular_crossover import AnularCrossover
from genetic_algorithms.src.hyperparams.crossover.one_point_crossover import OnePointCrossover
from genetic_algorithms.src.hyperparams.crossover.two_point_crossover import TwoPointCrossover
from genetic_algorithms.src.hyperparams.crossover.uniform_crossover import UniformCrossover
from genetic_algorithms.src.hyperparams.mutation.multi_gene_mutation import MultiGeneMutation
from genetic_algorithms.src.hyperparams.mutation.non_uniform_gene_mutation import NonUniformGeneMutation
from genetic_algorithms.src.hyperparams.mutation.single_gene_mutation import SingleGeneMutation
from genetic_algorithms.src.hyperparams.mutation.total_gene_mutation import TotalGeneMutation
from genetic_algorithms.src.hyperparams.replacement.fill_all_replacement import FillAllReplacement
from genetic_algorithms.src.hyperparams.replacement.fill_parent_replacement import FillParentReplacement
from genetic_algorithms.src.hyperparams.selection.boltzmann_selection import BoltzmannSelection
from genetic_algorithms.src.hyperparams.selection.combined_selection import CombinedSelection
from genetic_algorithms.src.hyperparams.selection.deterministic_tournament_selection import DeterministicTournamentSelection
from genetic_algorithms.src.hyperparams.selection.elite_selection import EliteSelection
from genetic_algorithms.src.hyperparams.selection.probabilistic_tournament_selection import ProbabilisticTournamentSelection
from genetic_algorithms.src.hyperparams.selection.ranking_selection import RankingSelection
from genetic_algorithms.src.hyperparams.selection.roulette_selection import RouletteSelection
from genetic_algorithms.src.hyperparams.selection.universal_selection import UniversalSelection
from genetic_algorithms.src.hyperparams.termination.gen_amount_termination import GenAmountTermination
from genetic_algorithms.src.hyperparams.termination.structural_stability_termination import StructuralStabilityTermination
from genetic_algorithms.src.hyperparams.termination.content_stability_termination import ContentStabilityTermination
from genetic_algorithms.src.hyperparams.termination.acceptable_solution_termination import AcceptableSolutionTermination
from genetic_algorithms.src.hyperparams.mutation.uniform_gene_mutation import UniformGeneMutation

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
    "two_point": TwoPointCrossover,
    "uniform": UniformCrossover,
    "anular": AnularCrossover,
}

REPLACEMENT_MAP = {
    "fill_all": FillAllReplacement,
    "fill_parent": FillParentReplacement,
}

MUTATION_MAP = {
    "single_gene": SingleGeneMutation,
    "multi_gene": MultiGeneMutation,
    "uniform_gene": UniformGeneMutation,
    "non_uniform_gene": NonUniformGeneMutation,
    "total_gene": TotalGeneMutation
}

TERMINATION_MAP = {
    "generation_amount": GenAmountTermination,
    "structural_stability": StructuralStabilityTermination,
    "content_stability": ContentStabilityTermination,
    "acceptable_solution": AcceptableSolutionTermination,
}
