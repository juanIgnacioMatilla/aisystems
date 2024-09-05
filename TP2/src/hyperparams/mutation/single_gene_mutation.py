from random import random, randint
from TP2.src.hyperparams.mutation.abstract_mutation import Mutation
from TP2.src.model.chromosome import Chromosome
from TP2.src.model.individual import Individual


class SingleGeneMutation(Mutation):
    def mutate(self, ind1: Individual) -> Individual:
        # Convert the NamedTuple to a list of attributes for mutation
        chromosome = list(ind1.chromosome)

        # Determine the index of the gene to mutate
        num_genes = len(chromosome)
        gene_to_mutate = randint(0, num_genes - 1)

        if random() < self.p_m:
            # Mutate the selected gene
            if gene_to_mutate == 0:
                mutation_amount = random() * 0.1  # Example mutation amount
                chromosome[gene_to_mutate] += mutation_amount
                # Constrain height to be within 1.3 and 2.0
                chromosome[gene_to_mutate] = min(max(chromosome[gene_to_mutate], 1.3), 2.0)
            else:
                # Mutate 'strength_points', 'agility_points', etc., which are ints
                current_value = chromosome[gene_to_mutate]
                # Apply mutation by incrementing or decrementing the value
                chromosome[gene_to_mutate] = max(0, current_value + randint(-5, 5))

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
