import random
from typing import Tuple, Dict, List

from TP2.src.hyperparams.crossover.abstract_crossover import Crossover
from TP2.src.model.chromosome import Chromosome
from TP2.src.model.individual import Individual


def perform_crossover(chromosome1: Tuple[int, ...], chromosome2: Tuple[int, ...], start_point: int, segment_length: int) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Perform the crossover operation."""
    num_attributes = len(chromosome1)
    offspring_values1 = list(chromosome1)
    offspring_values2 = list(chromosome2)

    for i in range(segment_length):
        index = (start_point + i) % num_attributes
        offspring_values1[index], offspring_values2[index] = offspring_values2[index], offspring_values1[index]

    return tuple(offspring_values1), tuple(offspring_values2)


class AnularCrossover(Crossover):
    def __init__(self):
        self.combination_list = []

    def generate_combinations(self, num_attributes: int) -> None:
        """Generate all possible combinations of start_point and segment_length."""
        self.combination_list = [(start_point, segment_length)
                                 for start_point in range(num_attributes)
                                 for segment_length in range(1, num_attributes)]

        # Shuffle the list to ensure random checking
        random.shuffle(self.combination_list)

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        chromosome1 = parent1.chromosome
        chromosome2 = parent2.chromosome

        # Define the number of attributes in the chromosome
        num_attributes = len(chromosome1._fields)

        # Define the total points for normalization
        max_points = parent1.chromosome.vigor_points + parent1.chromosome.constitution_points + parent1.chromosome.strength_points + parent1.chromosome.agility_points + parent1.chromosome.intelligence_points

        # Function to check if chromosome values are normalized
        def is_normalized(chromosome_values: Tuple[int, ...]) -> bool:
            return sum(chromosome_values[1:]) == max_points

        # Generate and shuffle all possible combinations
        self.generate_combinations(num_attributes)

        # Try all shuffled combinations until a valid offspring is found
        for start_point, segment_length in self.combination_list:
            offspring_values1, offspring_values2 = perform_crossover(
                tuple(chromosome1), tuple(chromosome2), start_point, segment_length)

            # Check if both offspring are normalized
            if is_normalized(offspring_values1) and is_normalized(offspring_values2):
                # Rebuild NamedTuples from values
                offspring_chromosome1 = Chromosome(*offspring_values1)
                offspring_chromosome2 = Chromosome(*offspring_values2)

                # Create new Individual instances with the offspring chromosomes
                offspring1 = Individual(parent1.type, offspring_chromosome1)
                offspring2 = Individual(parent2.type, offspring_chromosome2)

                return offspring1, offspring2

        # If no valid offspring are found after checking all combinations, return the parents
        return parent1, parent2
