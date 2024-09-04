from TP2.model.chromosome import Chromosome
from TP2.model.eve import EVE
from TP2.model.individual_types import IndividualTypes


class Individual:

    def __init__(self, type_ind: IndividualTypes, chromosome: Chromosome):
        self.chromosome = chromosome
        self.type = type_ind

    def fitness(self) -> float:
        return EVE.calculate_fitness(self.type, self.chromosome)
