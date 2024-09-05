import json

from TP2.src.hyperparams.crossover.one_point_crossover import OnePointCrossover
from TP2.src.hyperparams.hyperparams import Hyperparams
from TP2.src.hyperparams.mutation.single_gene_mutation import SingleGeneMutation
from TP2.src.hyperparams.replacement.fill_all_replacement import FillAllReplacement
from TP2.src.hyperparams.selection.elite_selection import EliteSelection
from TP2.src.hyperparams.termination.gen_amount_termination import GenAmountTermination
from TP2.src.model.individual_types import IndividualTypes
from TP2.src.genetic_engine import GeneticEngine
from TP2.utils.config_handler import get_strategies
from TP2.utils.hyperparams_mapping import SELECTION_MAP, CROSSOVER_MAP, MUTATION_MAP, REPLACEMENT_MAP, TERMINATION_MAP


def load_config(filename: str):
    """Load configuration from a JSON file."""
    with open(filename, 'r') as file:
        return json.load(file)


def main():
    # Load configuration from JSON
    config = load_config('config.json')

    # Iterate over each run configuration in the JSON
    for run_config in config:
        # Extract hyperparameters and instantiate strategy classes

        (selection_strategy,
         crossover_strategy,
         mutation_strategy,
         replacement_strategy,
         termination_strategy) = get_strategies(run_config)
        # Set up hyperparameters
        hyperparams: Hyperparams = {
            'selection_strategy': selection_strategy,
            'crossover_strategy': crossover_strategy,
            'mutation_strategy': mutation_strategy,
            'termination_strategy': termination_strategy,
            'replacement_strategy': replacement_strategy
        }

        # Extract other parameters
        ind_type = IndividualTypes[run_config['individual_type']]
        total_points = run_config['total_points']
        population_size = run_config['population_size']
        time_limit = run_config['time_limit']

        engine = GeneticEngine(hyperparams)
        for i in range(run_config['run']):
            (initial_population,
             final_population,
             generations,
             total_time,
             (best_ind,best_generation)) = engine.run(
                ind_type,
                total_points,
                population_size,
                time_limit
            )

            # Output the results
            print(f"Run {i + 1}/{run_config['run']}:")
            print(f"Initial Population: {sorted(initial_population, reverse=True)}")
            print(f"Final Population: {sorted(final_population, reverse=True)}")
            print(f"Generations: {generations}")
            print(f"Total Time: {total_time:.2f} seconds")
            print(f"Best individual in generation {best_generation}: {best_ind}\n")


if __name__ == "__main__":
    main()
