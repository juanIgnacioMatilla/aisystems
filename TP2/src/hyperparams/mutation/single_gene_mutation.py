from random import random, randint, choice
from TP2.src.hyperparams.mutation.abstract_mutation import Mutation
from TP2.src.model.attributes import Attributes
from TP2.src.model.chromosome import Chromosome
from TP2.src.model.individual import Individual


class SingleGeneMutation(Mutation):
    def mutate(self, ind1: Individual) -> Individual:
        # Determine the index of the gene to mutate
        chromosome = [ind1.chromosome.height] + list(ind1.chromosome.att_genes)
        locus = randint(0, len(chromosome) - 1)
        if random() < self.p_m:
            if locus == 0:
                # Mutate height
                mutation_amount = random() * 0.1 * choice([1, -1])
                chromosome[locus] += mutation_amount
                # Constrain height to be within 1.3 and 2.0
                chromosome[locus] = min(max(chromosome[locus], 1.3), 2.0)
            else:
                # Mutate other attributes: strength, agility, etc.
                current_value = chromosome[locus]
                eligible_values = [attr for attr in Attributes if attr != current_value]
                # Update the gene to be mutated
                chromosome[locus] = choice(eligible_values)
        return Individual(ind1.type, Chromosome(height=chromosome[0], att_genes=chromosome[1:]))
