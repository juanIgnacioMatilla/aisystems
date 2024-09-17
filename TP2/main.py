import copy
import json
import numpy as np
from TP2.src.hyperparams.hyperparams import Hyperparams
from TP2.src.model.individual_types import IndividualTypes
from TP2.src.genetic_engine import GeneticEngine
from TP2.utils.config_handler import get_strategies
from TP2.utils.ploter import plot_fitness_per_generation, plot_diversity_per_generation, \
    plot_chromosome_distribution


def load_config(filename: str):
    """Load configuration from a JSON file."""
    with open(filename, 'r') as file:
        return json.load(file)


def main():
    # Load configuration from JSON
    config = load_config('config.json')

    # Iterate over each run configuration in the JSON
    for run_config in config:
        run_config_cp = copy.deepcopy(run_config)
        # Extract hyperparameters and instantiate strategy classes
        total_diversity = []
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
        best_inds = []
        best_gens = []
        engine = GeneticEngine(hyperparams)
        for i in range(run_config['runs']):
            # Reset strategy inner states
            selection_strategy.reset()
            crossover_strategy.reset()
            mutation_strategy.reset()
            replacement_strategy.reset()
            termination_strategy.reset()

            (initial_population,
             final_population,
             generations,
             total_time,
             (best_ind, best_generation)) = engine.run(
                ind_type,
                total_points,
                population_size,
                time_limit
            )
            print(f"BEST {best_ind.type.value}")
            print(f"Fitness:{best_ind.fitness()}")
            print(f"Fuerza: {best_ind.chromosome.strength_points()}")
            print(f"Destreza: {best_ind.chromosome.dexterity_points()}")
            print(f"Inteligencia: {best_ind.chromosome.intelligence_points()}")
            print(f"Vigor: {best_ind.chromosome.vitality_points()}")
            print(f"Constitucion: {best_ind.chromosome.constitution_points()}")
            print(f"Altura: {best_ind.chromosome.height:.3f}")
            print(f"Tiempo: {total_time:.3f}")
            plot_chromosome_distribution(best_ind.chromosome, best_ind.type.value)


            print(f"Run {i + 1}/{run_config['runs']}:")

if __name__ == "__main__":
    main()
