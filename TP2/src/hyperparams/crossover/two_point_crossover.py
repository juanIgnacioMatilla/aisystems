from random import random, randint, choice
from typing import Tuple

from TP2.src.hyperparams.crossover.abstract_crossover import Crossover
from TP2.src.model.chromosome import Chromosome
from TP2.src.model.individual import Individual


class TwoPointCrossover(Crossover):

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        parent1_chromosome = [parent1.chromosome.height] + list(parent1.chromosome.att_genes)
        #parent1_chromosome = [parent1_chromosome[0]] + parent1_chromosome[1]
        parent2_chromosome = [parent2.chromosome.height] + list(parent2.chromosome.att_genes)
        #parent1_chromosome = [parent2_chromosome[0]] + parent2_chromosome[1]

        locus = randint(0, len(parent1_chromosome) - 1)
        locus2 = randint(locus, len(parent1_chromosome) - 1)


        child1_chromosome = parent1_chromosome[:locus] + parent2_chromosome[locus:locus2] + parent1_chromosome[locus2:]
        child2_chromosome = parent2_chromosome[:locus] + parent1_chromosome[locus:locus2] + parent2_chromosome[locus2:]

        child1 = Individual(type_ind= parent1.type, chromosome= Chromosome(height=child1_chromosome[0], att_genes=child1_chromosome[1:]))
        child2 = Individual(type_ind= parent2.type, chromosome= Chromosome(height=child2_chromosome[0], att_genes=child2_chromosome[1:]))

        return child1, child2