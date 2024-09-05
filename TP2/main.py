import json

from TP2.src.hyperparams.crossover.one_point_crossover import OnePointCrossover
from TP2.src.hyperparams.hyperparams import Hyperparams
from TP2.src.hyperparams.mutation.single_gene_mutation import SingleGeneMutation
from TP2.src.hyperparams.replacement.fill_all_replacement import FillAllReplacement
from TP2.src.hyperparams.selection.elite_selection import EliteSelection
from TP2.src.hyperparams.termination.gen_amount_termination import GenAmountTermination
from TP2.src.model.individual_types import IndividualTypes
from TP2.src.genetic_engine import GeneticEngine
from TP2.utils.hyperparams_mapping import SELECTION_MAP, CROSSOVER_MAP, MUTATION_MAP, REPLACEMENT_MAP, TERMINATION_MAP
import pandas as pd


def main():
    # Set up hyperparameters and strategies
    hyperparams: Hyperparams = {
        'selection_strategy': EliteSelection(k=5),
        'crossover_strategy': OnePointCrossover(),
        'mutation_strategy': SingleGeneMutation(0.2),
        'termination_strategy': GenAmountTermination(50),
        'replacement_strategy': FillAllReplacement()
    }

    ind_type = IndividualTypes.WARRIOR
    total_points = 100
    population_size = 20
    time_limit = 60

    engine = GeneticEngine(hyperparams)

    initial_population, final_population, generations, total_time = engine.run(
        ind_type,
        total_points,
        population_size,
        time_limit
    )
    print(f"Initial Population: {sorted(initial_population)}")
    print(f"Final Population: {sorted(final_population)}")
    print(f"Generations: {generations}")
    print(f"Total Time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
