from abc import ABC
from random import random, randint, choice
from TP2.src.hyperparams.mutation.abstract_mutation import Mutation
from TP2.src.hyperparams.mutation.multi_gene_mutation import MultiGeneMutation
from TP2.src.hyperparams.mutation.single_gene_mutation import SingleGeneMutation
from TP2.src.model.chromosome import Chromosome
from TP2.src.model.individual import Individual

# Uniform Mutation Strategies

class UniformSingleGeneMutation(SingleGeneMutation):
    pass  # Inherits all behavior from SingleGeneMutation; mutation probability is constant

class UniformMultiGeneMutation(MultiGeneMutation):
    pass  # Inherits all behavior from MultiGeneMutation; mutation probability is constant

# Non-Uniform Mutation Strategies

# class NonUniformMutation(Mutation, ABC):
#     def __init__(self, initial_p_m: float, decay_rate: float):
#         """
#         :param initial_p_m: Initial probability of mutating a gene
#         :param decay_rate: Rate at which the mutation probability decreases over generations
#         """
#         super().__init__(initial_p_m)
#         self.decay_rate = decay_rate
#
#     def get_current_p_m(self, generation: int) -> float:
#         """Calculate the current probability of mutation based on the generation."""
#         return max(0.0, self.p_m * (1 - self.decay_rate * generation))
#
# class NonUniformSingleGeneMutation(NonUniformMutation):
#     def mutate(self, ind1: Individual, generation: int) -> Individual:
#         # Update mutation probability based on the generation
#         self.p_m = self.get_current_p_m(generation)
#         return super().mutate(ind1)
#
# class NonUniformMultiGeneMutation(NonUniformMutation):
#     def mutate(self, ind1: Individual, generation: int) -> Individual:
#         # Update mutation probability based on the generation
#         self.p_m = self.get_current_p_m(generation)
#         return super().mutate(ind1)
