import random
from typing import List

from genetic_algorithms.src.model.attributes import Attributes
from genetic_algorithms.src.model.chromosome import Chromosome
from genetic_algorithms.src.model.individual import Individual
from genetic_algorithms.src.model.individual_types import IndividualTypes


def generate_initial_population(
        ind_type: IndividualTypes,
        total_points: int,
        population_size: int
) -> List[Individual]:
    population = []

    for _ in range(population_size):
        # Generate a random distribution of total_points across the attributes

        att_array : List[Attributes] = []
        # Initialize a Chromosome instance
        for  i in range(total_points):
            att_array.append(random.choice(list(Attributes)))

        chromosome = Chromosome(
            height=random.uniform(1.3, 2.0),
            att_genes=att_array
        )

        # Create an Individual with the type and chromosome
        individual = Individual(ind_type, chromosome)
        population.append(individual)

    return population
