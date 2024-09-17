import time
from typing import List

import numpy as np

from TP2.src.hyperparams.hyperparams import Hyperparams
from TP2.src.model.diversity import Diversity
from TP2.src.model.individual_types import IndividualTypes
from TP2.src.model.individual import Individual
from TP2.src.model.initial_population import generate_initial_population


class GeneticEngine:
    def __init__(self, hyperparams: Hyperparams):
        self.selection_strategy = hyperparams['selection_strategy']
        self.crossover_strategy = hyperparams['crossover_strategy']
        self.mutation_strategy = hyperparams['mutation_strategy']
        self.termination_strategy = hyperparams['termination_strategy']
        self.replacement_strategy = hyperparams['replacement_strategy']
        self.generation = None

    def run(self, ind_type: IndividualTypes, total_points: int, population_size: int, time_limit: float):
        # self.fitness_history = []
        # self.diversity_history = []
        if time_limit < 10 or time_limit > 120:
            raise ValueError("time limit must be between 10 and 120")
        population = generate_initial_population(ind_type, total_points, population_size)
        initial_population = population
        # self.fitness_history.append(self._calculate_fitness_mean(population))
        # self.diversity_history.append(Diversity.calculate_diversity(population))
        self.generation = 0
        start_time = time.time()
        # Initialize the best individual tracking
        best_individual = min(population, key=lambda ind: ind.fitness())
        best_generation = self.generation

        while not self.termination_strategy.should_terminate(population, self.generation) and (
                time.time() - start_time) < time_limit:
            parents = self.selection_strategy.select(population)
            offspring = self._generate_offspring(parents)
            offspring = self._mutate_offspring(offspring)
            population = self.replacement_strategy.replace(population, offspring)
            # self.fitness_history.append(self._calculate_fitness_mean(population))
            # if(self.generation % 25 == 0):
            #     self.diversity_history.append(Diversity.calculate_diversity(population))
            self.generation += 1
            # Update the best individual and its generation if a better one is found
            current_best = min(population, key=lambda ind: ind.fitness())
            if current_best.fitness() > best_individual.fitness():
                best_individual = current_best
                best_generation = self.generation
        total_time = time.time() - start_time
        return initial_population, population, self.generation, total_time, (best_individual, best_generation)

    def _calculate_fitness_mean(self, population):
        final_fitness_values = [individual.fitness() for individual in population]
        final_fitness_mean = np.mean(final_fitness_values)
        final_fitness_std = np.std(final_fitness_values)
        return final_fitness_mean, final_fitness_std

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
