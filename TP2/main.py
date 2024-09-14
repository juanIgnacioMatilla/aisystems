import json
import numpy as np
from TP2.src.hyperparams.hyperparams import Hyperparams
from TP2.src.model.individual_types import IndividualTypes
from TP2.src.genetic_engine import GeneticEngine
from TP2.utils.config_handler import get_strategies


def load_config(filename: str):
    """Load configuration from a JSON file."""
    with open(filename, 'r') as file:
        return json.load(file)


def main():
    # Load configuration from JSON
    config = load_config('config.json')

    # List to store results of each run
    results = []

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

            initial_fitness_values = [individual.fitness() for individual in initial_population]
            initial_fitness_mean = np.mean(initial_fitness_values)
            initial_fitness_std = np.std(initial_fitness_values)

            final_fitness_values = [individual.fitness() for individual in final_population]
            final_fitness_mean = np.mean(final_fitness_values)
            final_fitness_std = np.std(final_fitness_values)

            # Store the results of the run
            run_result = {
                'run': i + 1,
                'initial_fitness_mean': initial_fitness_mean,
                'initial_fitness_std': initial_fitness_std,
                'final_fitness_mean': final_fitness_mean,
                'final_fitness_std': final_fitness_std,
                'generations': generations,
                'total_time': total_time,
                'best_individual': best_ind,
                'best_generation': best_generation,
                'termination_strategy': termination_strategy.__class__.__name__
            }
            results.append(run_result)

            # Output the results
            print(f"Termination strategy: {termination_strategy.__class__.__name__}")
            print(f"Run {i + 1}/{run_config['runs']}:")
            print(f"Initial Population fitness mean: {initial_fitness_mean:.4f} +- {initial_fitness_std:.4f}")
            print(f"Final Population fitness mean:  {final_fitness_mean:.4f} +- {final_fitness_std:.4f}")
            print(f"Generations: {generations}")
            print(f"Total Time: {total_time:.2f} seconds")
            print(f"Best individual in generation {best_generation}: {best_ind}")
            print(f"Params for best individual:")
            print(f"Individual Type: {ind_type.value}")
            print(f"Total Points: {total_points}")
            print(f"Strength points: {best_ind.chromosome.strength_points()}")
            print(f"Agility points: {best_ind.chromosome.dexterity_points()}")
            print(f"Vigor points: {best_ind.chromosome.vitality_points()}")
            print(f"Constitution points: {best_ind.chromosome.constitution_points()}")
            print(f"Intelligence points: {best_ind.chromosome.intelligence_points()}")
            print(f"Height: {best_ind.chromosome.height:.3f}\n")

    initial_fitness_means = [result['initial_fitness_mean'] for result in results]
    initial_fitness_stds = [result['initial_fitness_std'] for result in results]
    final_fitness_means = [result['final_fitness_mean'] for result in results]
    final_fitness_stds = [result['final_fitness_std'] for result in results]
    generations = [result['generations'] for result in results]
    total_times = [result['total_time'] for result in results]

    overall_results = {
        'initial_fitness_mean': np.mean(initial_fitness_means),
        'initial_fitness_std': np.std(initial_fitness_means),
        'final_fitness_mean': np.mean(final_fitness_means),
        'final_fitness_std': np.std(final_fitness_means),
        'generations_mean': np.mean(generations),
        'generations_std': np.std(generations),
        'total_time_mean': np.mean(total_times),
        'total_time_std': np.std(total_times)
    }

    print(f"Overall Results: {overall_results}")

    return results


if __name__ == "__main__":
    main()