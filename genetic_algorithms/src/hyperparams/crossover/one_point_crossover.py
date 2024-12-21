import random
from typing import Tuple, List

from genetic_algorithms.src.hyperparams.crossover.abstract_crossover import Crossover
from genetic_algorithms.src.model.chromosome import Chromosome
from genetic_algorithms.src.model.individual import Individual


class OnePointCrossover(Crossover):

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        parent1_chromosome = [parent1.chromosome.height] + list(parent1.chromosome.att_genes)
        #parent1_chromosome = [parent1_chromosome[0]] + parent1_chromosome[1]
        parent2_chromosome = [parent2.chromosome.height] + list(parent2.chromosome.att_genes)
        #parent1_chromosome = [parent2_chromosome[0]] + parent2_chromosome[1]

        locus = random.randint(0, len(parent1_chromosome) - 1)

        child1_chromosome = parent1_chromosome[:locus] + parent2_chromosome[locus:]
        child2_chromosome = parent2_chromosome[:locus] + parent1_chromosome[locus:]

        child1 = Individual(type_ind= parent1.type, chromosome= Chromosome(height=child1_chromosome[0], att_genes=child1_chromosome[1:]))
        child2 = Individual(type_ind= parent2.type, chromosome= Chromosome(height=child2_chromosome[0], att_genes=child2_chromosome[1:]))

        return child1, child2