from random import randint
from typing import Tuple
import numpy as np

from TP2.src.hyperparams.crossover.abstract_crossover import Crossover
from TP2.src.model.chromosome import Chromosome
from TP2.src.model.individual import Individual


class AnularCrossover(Crossover):
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        parent1_chromosome = [parent1.chromosome.height] + list(parent1.chromosome.att_genes)
        parent2_chromosome = [parent2.chromosome.height] + list(parent2.chromosome.att_genes)

        chromosome_length = len(parent1_chromosome) - 1

        locus = randint(0, chromosome_length)
        length = randint(0, int(np.ceil(chromosome_length / 2)) )

        child1_chromosome = []
        child2_chromosome = []

        if locus + length > chromosome_length:
            child1_chromosome = parent2_chromosome[:(locus + length) % chromosome_length] + parent1_chromosome[(locus + length) % chromosome_length:locus] + parent2_chromosome[locus:]
            child2_chromosome = parent1_chromosome[:(locus + length) % chromosome_length] + parent2_chromosome[(locus + length) % chromosome_length:locus] + parent1_chromosome[locus:]
        else:
            child1_chromosome = parent1_chromosome[:locus] + parent2_chromosome[locus:locus + length] + parent1_chromosome[locus + length:]
            child2_chromosome = parent2_chromosome[:locus] + parent1_chromosome[locus:locus + length] + parent2_chromosome[locus  + length:]


        child1 = Individual(type_ind= parent1.type, chromosome= Chromosome(height=child1_chromosome[0], att_genes=child1_chromosome[1:]))
        child2 = Individual(type_ind= parent2.type, chromosome= Chromosome(height= child2_chromosome[0], att_genes= child2_chromosome[1:]))

        return child1, child2