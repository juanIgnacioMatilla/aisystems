import time
from typing import List
from genetic_algorithms.src.hyperparams.hyperparams import Hyperparams
from genetic_algorithms.src.model.individual_types import IndividualTypes
from genetic_algorithms.src.model.individual import Individual
from genetic_algorithms.src.model.initial_population import generate_initial_population


class GeneticEngine:
    def __init__(self, hyperparams: Hyperparams):
        self.selection_strategy = hyperparams['selection_strategy']
        self.crossover_strategy = hyperparams['crossover_strategy']
        self.mutation_strategy = hyperparams['mutation_strategy']
        self.termination_strategy = hyperparams['termination_strategy']
        self.replacement_strategy = hyperparams['replacement_strategy']
        self.generation = None

    def run(self, ind_type: IndividualTypes, total_points: int, population_size: int, time_limit: float):
        if time_limit < 10 or time_limit > 120:
            raise ValueError("time limit must be between 10 and 120")
        population = generate_initial_population(ind_type, total_points, population_size)
        initial_population = population
        self.generation = 0
        start_time = time.time()

        # Initialize the best individual tracking
        best_individual = max(population, key=lambda ind: ind.fitness())
        best_generation = self.generation

        while not self.termination_strategy.should_terminate(population, self.generation) and (
                time.time() - start_time) < time_limit:
            parents = self.selection_strategy.select(population)
            offspring = self._generate_offspring(parents)
            offspring = self._mutate_offspring(offspring)
            population = self.replacement_strategy.replace(population, offspring)
            self.generation += 1
            # Update the best individual and its generation if a better one is found
            current_best = max(population, key=lambda ind: ind.fitness())
            if current_best.fitness() > best_individual.fitness():
                best_individual = current_best
                best_generation = self.generation

        total_time = time.time() - start_time
        return initial_population, population, self.generation, total_time, (best_individual, best_generation)

    def _generate_offspring(self, parents: List[Individual]) -> List[Individual]:
        offspring: List[Individual] = []
        num_parents = len(parents)
        for i in range(0, num_parents, 2):
            child1, child2 = self.crossover_strategy.crossover(parents[i], parents[(i + 1) % num_parents])
            offspring.extend([child1, child2])
        return offspring

    def _mutate_offspring(self, offspring: List[Individual]) -> List[Individual]:
        offspring = [self.mutation_strategy.mutate(individual, self.generation) for individual in offspring]
        return offspring
