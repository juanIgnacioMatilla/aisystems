from TP2.model.genes import Genes
from TP2.model.eve import EVE


class Individual:

    def __init__(self, genes: Genes):
        strength_points = genes['strength_points']
        agility_points = genes['agility_points']
        intelligence_points = genes['intelligence_points']
        vigor_points = genes['vigor_points']
        constitution_points = genes['constitution_points']

        self.type = genes['type']
        self.height = genes['height']
        # Calculate attributes using EVE class
        self.force_total = EVE.calculate_force_total(strength_points)
        self.agility_total = EVE.calculate_agility_total(agility_points)
        self.intelligence_total = EVE.calculate_intelligence_total(intelligence_points)
        self.vigor_total = EVE.calculate_vigor_total(vigor_points)
        self.constitution_total = EVE.calculate_constitution_total(constitution_points)
        self.atm = EVE.calculate_atm(self.height)
        self.dem = EVE.calculate_dem(self.height)

    def fitness(self) -> float:
        return EVE.calculate_fitness(self)
