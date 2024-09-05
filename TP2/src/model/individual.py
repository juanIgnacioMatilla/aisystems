from TP2.src.model.chromosome import Chromosome
from TP2.src.model.eve import EVE
from TP2.src.model.individual_types import IndividualTypes


class Individual:

    def __init__(self, type_ind: IndividualTypes, chromosome: Chromosome):
        self.chromosome = chromosome
        self.type = type_ind

    def fitness(self) -> float:
        return EVE.calculate_fitness(self.type, self.chromosome)

    def __repr__(self) -> str:
        # String representation of the object, displaying type and chromosome details
        return (f"Individual(fitness: {self.fitness():.2f})"
                # f"{self.type}, "
                # f"chromosome=height: {self.chromosome.height}, "
                # f"strength_points: {self.chromosome.strength_points}, "
                # f"agility_points: {self.chromosome.agility_points}, "
                # f"intelligence_points: {self.chromosome.intelligence_points}, "
                # f"vigor_points: {self.chromosome.vigor_points}, "
                # f"constitution_points: {self.chromosome.constitution_points}"
                )

    def __lt__(self, other: "Individual") -> bool:
        if not isinstance(other, Individual):
            return NotImplemented
        return self.fitness() < other.fitness()

    def __gt__(self, other: "Individual") -> bool:
        if not isinstance(other, Individual):
            return NotImplemented
        return self.fitness() > other.fitness()





