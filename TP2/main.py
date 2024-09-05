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

        # Initialize Selection Strategy
        selection_params = run_config['hyperparams']['selection']
        selection_class = SELECTION_MAP[selection_params.pop('name')]
        selection_strategy = selection_class(**selection_params)

        # Initialize Crossover Strategy (no params needed)
        crossover_class = CROSSOVER_MAP[run_config['hyperparams']['crossover']['name']]
        crossover_strategy = crossover_class()

        # Initialize Mutation Strategy
        mutation_params = run_config['hyperparams']['mutation']
        mutation_class = MUTATION_MAP[mutation_params.pop('name')]
        mutation_strategy = mutation_class(**mutation_params)

        # Initialize Replacement Strategy (no params needed)
        replacement_class = REPLACEMENT_MAP[run_config['hyperparams']['replacement']['name']]
        replacement_strategy = replacement_class()

        # Initialize Termination Strategy
        termination_params = run_config['hyperparams']['termination']
        termination_class = TERMINATION_MAP[termination_params.pop('name')]
        termination_strategy = termination_class(**termination_params)

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

        # Initialize and run the genetic engine
        engine = GeneticEngine(hyperparams)
        initial_population, final_population, generations, total_time = engine.run(
            ind_type,
            total_points,
            population_size,
            time_limit
        )

        # Output the results
        print(f"Initial Population: {sorted(initial_population)}")
        print(f"Final Population: {sorted(final_population)}")
        print(f"Generations: {generations}")
        print(f"Total Time: {total_time:.2f} seconds")
if __name__ == "__main__":
    main()
