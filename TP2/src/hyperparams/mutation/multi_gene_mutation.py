from random import random, randint, choice
from TP2.src.hyperparams.mutation.abstract_mutation import Mutation
from TP2.src.model.chromosome import Chromosome
from TP2.src.model.individual import Individual


class MultiGeneMutation(Mutation):
    def mutate(self, ind1: Individual) -> Individual:
        # Convert the NamedTuple to a list of attributes for mutation
        chromosome = list(ind1.chromosome)
        num_genes = len(chromosome)

        # Iterate over each gene and determine if it should mutate
        for i in range(num_genes):
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
                    random_mutation = randint(-5, 5)

                    # Ensure the mutated value does not exceed 150
                    if current_value + random_mutation > 150:
                        random_mutation = 150 - current_value

                    # Update the gene to be mutated
                    chromosome[i] = max(0, current_value + random_mutation)

                    # Calculate the change (delta)
                    delta1 = abs(current_value - chromosome[i])

                    # Find another gene to balance the change
                    other_gene_to_mutate = randint(1, num_genes - 1)
                    while other_gene_to_mutate == i or \
                            chromosome[other_gene_to_mutate] + delta1 > 150 or \
                            chromosome[other_gene_to_mutate] - delta1 < 0:
                        other_gene_to_mutate = randint(1, num_genes - 1)

                    # Balance the other gene by applying the delta
                    if chromosome[i] > current_value:
                        chromosome[other_gene_to_mutate] = chromosome[other_gene_to_mutate] - delta1
                    else:
                        chromosome[other_gene_to_mutate] = chromosome[other_gene_to_mutate] + delta1

        # Create a new Chromosome with the mutated values
        mutated_chromosome = Chromosome(
            height=chromosome[0],
            strength_points=chromosome[1],
            agility_points=chromosome[2],
            intelligence_points=chromosome[3],
            vigor_points=chromosome[4],
            constitution_points=chromosome[5]
        )

        # Return a new Individual with the mutated chromosome
        mutated_individual = Individual(ind1.type, mutated_chromosome)
        return mutated_individual
