import time
import random
from typing import List
from TP2.hyperparams.hyperparams import Hyperparams
from TP2.model.chromosome import Chromosome
from TP2.model.individual_types import IndividualTypes
from model.individual import Individual


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
        generation = 0
        total_time = time.time()
        while not self.termination_strategy.should_terminate(population, generation) and total_time < time_limit:
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


def generate_initial_population(
        ind_type: IndividualTypes,
        total_points: int,
        population_size: int
) -> List[Individual]:
    population = []

    for _ in range(population_size):
        # Generate a random distribution of total_points across the attributes
        points_distribution = _distribute_points(total_points, 5)

        chromosome: Chromosome = {
            'height': random.uniform(1.3, 2.0),
            'strength_points': points_distribution[0],
            'agility_points': points_distribution[1],
            'intelligence_points': points_distribution[2],
            'vigor_points': points_distribution[3],
            'constitution_points': points_distribution[4],
        }

        # Create an Individual with the type and chromosome
        individual = Individual(ind_type, chromosome)
        population.append(individual)

    return population


def _distribute_points(total_points: int, num_attributes: int) -> List[int]:
    """Distributes total_points randomly across num_attributes attributes."""
    points = [0] * num_attributes
    for _ in range(total_points):
        # Randomly assign a point to one of the attributes
        points[random.randint(0, num_attributes - 1)] += 1
    return points
