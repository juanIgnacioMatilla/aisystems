import time
from typing import List
from TP2.src.hyperparams.hyperparams import Hyperparams
from TP2.src.model.chromosome import Chromosome, normalize_chromosome
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

    def run(self, ind_type: IndividualTypes, total_points: int, population_size: int, time_limit: float):
        if time_limit < 10 or time_limit > 120:
            raise ValueError("time limit must be between 10 and 120")
        population = generate_initial_population(ind_type, total_points, population_size)
        initial_population = population
        generation = 0
        start_time = time.time()

        while not self.termination_strategy.should_terminate(population, generation) and (
                time.time() - start_time) < time_limit:
            parents = self.selection_strategy.select(population)
            offspring = self._generate_offspring(parents)
            self._mutate_offspring(offspring)
            # Normalize in case after crossover and mutation any offspring violates the constraints on points or height
            for i in range(len(offspring)):
                offspring[i].chromosome = normalize_chromosome(offspring[i].chromosome, total_points)
            population = self.replacement_strategy.replace(population, offspring)
            generation += 1

        total_time = time.time() - start_time
        return initial_population,population, generation, total_time

    def _generate_offspring(self, parents: List[Individual]) -> List[Individual]:
        offspring: List[Individual] = []
        num_parents = len(parents)
        for i in range(0, num_parents, 2):
            child1, child2 = self.crossover_strategy.crossover(parents[i], parents[(i + 1) % num_parents])
            offspring.extend([child1, child2])
        return offspring

    def _mutate_offspring(self, offspring: List[Individual]):
        for individual in offspring:
            self.mutation_strategy.mutate(individual)
