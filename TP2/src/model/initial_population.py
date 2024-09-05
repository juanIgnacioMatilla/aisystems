import random
from typing import List

from TP2.src.model.chromosome import distribute_points, Chromosome
from TP2.src.model.individual import Individual
from TP2.src.model.individual_types import IndividualTypes


def generate_initial_population(
        ind_type: IndividualTypes,
        total_points: int,
        population_size: int
) -> List[Individual]:
    population = []

    for _ in range(population_size):
        # Generate a random distribution of total_points across the attributes
        points_distribution = distribute_points(total_points, 5)

        # Initialize a Chromosome instance
        chromosome = Chromosome(
            height=random.uniform(1.3, 2.0),
            strength_points=points_distribution[0],
            agility_points=points_distribution[1],
            intelligence_points=points_distribution[2],
            vigor_points=points_distribution[3],
            constitution_points=points_distribution[4]
        )

        # Create an Individual with the type and chromosome
        individual = Individual(ind_type, chromosome)
        population.append(individual)

    return population
