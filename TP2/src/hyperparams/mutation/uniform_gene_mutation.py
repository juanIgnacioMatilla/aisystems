from abc import ABC
from random import random, randint, choice
from TP2.src.hyperparams.mutation.abstract_mutation import Mutation
from TP2.src.hyperparams.mutation.multi_gene_mutation import MultiGeneMutation
from TP2.src.hyperparams.mutation.single_gene_mutation import SingleGeneMutation
from TP2.src.model.attributes import Attributes
from TP2.src.model.chromosome import Chromosome
from TP2.src.model.individual import Individual

# Uniform Mutation Strategies

class UniformGeneMutation(Mutation):
    def mutate(self, ind1: Individual) -> Individual:
        chromosome = [ind1.chromosome.height] + list(ind1.chromosome.att_genes)
        for i in range(len(chromosome)):
            if random() < self.p_m:
                if i == 0:
                    # Mutate height
                    mutation_amount = random() * 0.1 * choice([1, -1])
                    chromosome[i] += mutation_amount
                    # Constrain height to be within 1.3 and 2.0
                    chromosome[i] = min(max(chromosome[i], 1.3), 2.0)
                else:
                    # Mutate other attributes: strength, agility, etc.
                    current_value = chromosome[i]
                    eligible_values = [attr for attr in Attributes if attr != current_value]
                    # Update the gene to be mutated
                    chromosome[i] = choice(eligible_values)
        return Individual(ind1.type, Chromosome(height=chromosome[0], att_genes=chromosome[1:]))