import random
from itertools import combinations
from typing import Tuple

from TP2.src.hyperparams.crossover.abstract_crossover import Crossover
from TP2.src.model.chromosome import Chromosome
from TP2.src.model.individual import Individual


class TwoPointCrossover(Crossover):
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        chromosome1 = parent1.chromosome
        chromosome2 = parent2.chromosome

        # Define the number of attributes in the chromosome
        num_attributes = len(chromosome1._fields)  # Using NamedTuple fields count

        # Define the total points for normalization
        max_points = parent1.chromosome.vigor_points + parent1.chromosome.constitution_points + parent1.chromosome.strength_points + parent1.chromosome.agility_points + parent1.chromosome.intelligence_points

        # Function to check if chromosome values are normalized
        def is_normalized(chromosome_values: Tuple[int, ...]) -> bool:
            return sum(chromosome_values[1:]) == max_points

        # Array of valid crossover points = 0 for all positions (invalid)
        valid_crossover_combinations = [0] * (num_attributes * (num_attributes - 1) // 2)  # There are (n choose 2) combinations

        # Loop until valid offspring are found or all combinations are exhausted
        while sum(valid_crossover_combinations) < len(valid_crossover_combinations):
            # Select two distinct random crossover points
            point1, point2 = sorted(random.sample(range(0, num_attributes), 2))

            # Mark this combination as used
            index = list(combinations(range(0, num_attributes), 2)).index((point1, point2))
            valid_crossover_combinations[index] = 1

            # Create offspring chromosomes by swapping segments between the crossover points
            offspring_values1 = (
                *chromosome1[:point1],
                *chromosome2[point1:point2],
                *chromosome1[point2:]
            )
            offspring_values2 = (
                *chromosome2[:point1],
                *chromosome1[point1:point2],
                *chromosome2[point2:]
            )

            # Check if both offspring are normalized
            if is_normalized(offspring_values1) and is_normalized(offspring_values2):
                # Rebuild NamedTuples from values
                offspring_chromosome1 = Chromosome(*offspring_values1)
                offspring_chromosome2 = Chromosome(*offspring_values2)

                # Create new Individual instances with the offspring chromosomes
                offspring1 = Individual(parent1.type, offspring_chromosome1)
                offspring2 = Individual(parent2.type, offspring_chromosome2)

                return offspring1, offspring2

        # If no valid combination is found after exhausting all options, return the parents
        return parent1, parent2
