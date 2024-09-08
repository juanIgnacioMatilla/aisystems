import random
from itertools import product
from typing import Tuple

from TP2.src.hyperparams.crossover.abstract_crossover import Crossover
from TP2.src.model.chromosome import Chromosome
from TP2.src.model.individual import Individual


class UniformCrossover(Crossover):
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        chromosome1 = parent1.chromosome
        chromosome2 = parent2.chromosome

        # Define the number of attributes in the chromosome
        num_attributes = len(chromosome1._fields)  # Using NamedTuple fields count

        # Define the total points for normalization
        max_points = (
            parent1.chromosome.vigor_points
            + parent1.chromosome.constitution_points
            + parent1.chromosome.strength_points
            + parent1.chromosome.agility_points
            + parent1.chromosome.intelligence_points
        )

        # Function to check if chromosome values are normalized
        def is_normalized(chromosome_values: Tuple[int, ...]) -> bool:
            return sum(chromosome_values[1:]) == max_points

        # Generate all possible combinations of selecting genes from parent1 or parent2
        selection_patterns = list(product([0, 1], repeat=num_attributes))

        # Shuffle the selection patterns to introduce randomness
        random.shuffle(selection_patterns)

        # Iterate over the shuffled selection patterns
        for selection_pattern in selection_patterns:
            offspring_values1 = []
            offspring_values2 = []

            # Based on the selection pattern, choose genes from either parent1 or parent2
            for i in range(num_attributes):
                if selection_pattern[i] == 0:
                    offspring_values1.append(chromosome1[i])
                    offspring_values2.append(chromosome2[i])
                else:
                    offspring_values1.append(chromosome2[i])
                    offspring_values2.append(chromosome1[i])

            # Check if both offspring are normalized
            if is_normalized(tuple(offspring_values1)) and is_normalized(tuple(offspring_values2)):
                # Rebuild NamedTuples from values
                offspring_chromosome1 = Chromosome(*offspring_values1)
                offspring_chromosome2 = Chromosome(*offspring_values2)

                # Create new Individual instances with the offspring chromosomes
                offspring1 = Individual(parent1.type, offspring_chromosome1)
                offspring2 = Individual(parent2.type, offspring_chromosome2)

                return offspring1, offspring2

        # If no valid offspring are found after checking all combinations, return the parents
        return parent1, parent2
