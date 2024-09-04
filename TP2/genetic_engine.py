import time
from typing import List

from TP2.hyperparams.hyperparams import Hyperparams
from model.individual import Individual


class GeneticEngine:
    def __init__(self, hyperparams: Hyperparams, time_limit: int):
        if time_limit < 10 or time_limit > 120:
            raise ValueError("time limit must be between 10 and 120")
        self.time_limit = time_limit
        self.selection_strategy = hyperparams['selection_strategy']
        self.crossover_strategy = hyperparams['crossover_strategy']
        self.mutation_strategy = hyperparams['mutation_strategy']
        self.termination_strategy = hyperparams['termination_strategy']
        self.replacement_strategy = hyperparams['replacement_strategy']

    def generate_inital_population(self, type: str, total_points: int) -> List[Individual]:
        return []

    def run(self, population):
        generation = 0
        total_time = time.time()
        while not self.termination_strategy.should_terminate(population, generation) and total_time < self.time_limit:
            parents = self.selection_strategy.select(population)
            offspring = self._generate_offspring(parents)
            self._mutate_offspring(offspring)
            population = self.replacement_strategy.replace(population, offspring)
            generation += 1
            aux_time = time.time()
            total_time = aux_time - total_time
        return population, generation

    def _generate_offspring(self, parents: List[Individual]) -> List[Individual]:
        offspring: List[Individual] = []
        num_parents = len(parents)
        for i in range(num_parents):
            for j in range(i + 1, num_parents):
                child1, child2 = self.crossover_strategy.crossover(parents[i], parents[j])
                offspring.extend([child1, child2])
        return offspring

    def _mutate_offspring(self, offspring):
        for individual in offspring:
            self.mutation_strategy.mutate(individual)
