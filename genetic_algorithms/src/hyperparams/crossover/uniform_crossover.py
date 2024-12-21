from typing import Tuple

from genetic_algorithms.src.hyperparams.crossover.abstract_crossover import Crossover
from genetic_algorithms.src.model.chromosome import Chromosome
from genetic_algorithms.src.model.individual import Individual
import numpy as np

class UniformCrossover(Crossover):
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        parent1_chromosome = [parent1.chromosome.height] + list(parent1.chromosome.att_genes)
        parent2_chromosome = [parent2.chromosome.height] + list(parent2.chromosome.att_genes)


        randoms = np.random.rand(len(parent1_chromosome))
        child1_chromosome = []
        child2_chromosome = []
        for i in range(len(parent1_chromosome)):

            if randoms[i] < 0.5:
                child1_chromosome.append(parent1_chromosome[i])
                child2_chromosome.append(parent2_chromosome[i])
            else:
                child1_chromosome.append(parent2_chromosome[i])
                child2_chromosome.append(parent1_chromosome[i])

        child1 = Individual(type_ind=parent1.type,
                            chromosome=Chromosome(height=child1_chromosome[0], att_genes=child1_chromosome[1:]))
        child2 = Individual(type_ind=parent2.type,
                            chromosome=Chromosome(height=child2_chromosome[0], att_genes=child2_chromosome[1:]))

        return child1, child2

