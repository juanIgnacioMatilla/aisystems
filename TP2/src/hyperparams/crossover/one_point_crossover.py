import random
from typing import Tuple

from TP2.src.hyperparams.crossover.abstract_crossover import Crossover
from TP2.src.model.chromosome import Chromosome
from TP2.src.model.individual import Individual


class OnePointCrossover(Crossover):

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        chromosome1 = parent1.chromosome
        chromosome2 = parent2.chromosome

        # Define the number of attributes in the chromosome
        num_attributes = len(chromosome1._fields)  # Using NamedTuple fields count

        #Array of valid crossover points = 0 for all positions. Length is the number of attributes
        valid_crossover_points = [0]*num_attributes

        # Choose a crossover point between 1 and num_attributes - 1
        crossover_point = random.randint(1, num_attributes - 1)

        # Create offspring chromosomes by swapping segments after the crossover point
        offspring_values1 = (
            *chromosome1[:crossover_point],  # values from parent1 up to crossover point
            *chromosome2[crossover_point:]  # values from parent2 after crossover point
        )
        offspring_values2 = (
            *chromosome2[:crossover_point],  # values from parent2 up to crossover point
            *chromosome1[crossover_point:]  # values from parent1 after crossover point
        )

        # Verify that the offspring values are within the valid range of total points
        total_points1 = sum(offspring_values1[1:])
        total_points2 = sum(offspring_values2[1:])
        max_points = parent1.chromosome.vigor_points + parent1.chromosome.constitution_points + parent1.chromosome.strength_points + parent1.chromosome.agility_points + parent1.chromosome.intelligence_points
        while total_points1 != max_points or total_points2 != max_points:
            # If the offspring values do not match the target, choose a different crossover point
            valid_crossover_points[crossover_point] = 1
            crossover_point = random.randint(1, num_attributes - 1)
            while valid_crossover_points[crossover_point] == 1:
                crossover_point = random.randint(1, num_attributes - 1)
                if sum(valid_crossover_points) == num_attributes - 1:
                    return parent1, parent2

            offspring_values1 = (
                *chromosome1[:crossover_point],  # values from parent1 up to crossover point
                *chromosome2[crossover_point:]  # values from parent2 after crossover point
            )
            offspring_values2 = (
                *chromosome2[:crossover_point],  # values from parent2 up to crossover point
                *chromosome1[crossover_point:]  # values from parent1 after crossover point
            )
            total_points1 = sum(offspring_values1[1:])
            total_points2 = sum(offspring_values2[1:])

        # Rebuild NamedTuples from values
        offspring_chromosome1 = Chromosome(*offspring_values1)
        offspring_chromosome2 = Chromosome(*offspring_values2)

        # Create new Individual instances with the offspring chromosomes
        offspring1 = Individual(parent1.type, offspring_chromosome1)
        offspring2 = Individual(parent2.type, offspring_chromosome2)

        return offspring1, offspring2
