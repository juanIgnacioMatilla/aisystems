import random
from typing import NamedTuple, List


class Chromosome(NamedTuple):
    height: float
    strength_points: int
    agility_points: int
    intelligence_points: int
    vigor_points: int
    constitution_points: int

#TODO remove after finishing all hyperparams
def normalize_chromosome(chromosome: Chromosome, target_total) -> Chromosome:
    # Adjust height to be within 1.3 and 2.0
    height = min(max(chromosome.height, 1.3), 2.0)

    def strength_points(self):
        l = [1 for att in self.att_genes if att == Attributes.STR]
        return sum(l)

    def dexterity_points(self):
        l = [1 for att in self.att_genes if att == Attributes.DEX]
        return sum(l)

    def intelligence_points(self):
        l = [1 for att in self.att_genes if att == Attributes.INT]
        return sum(l)

    def constitution_points(self):
        l = [1 for att in self.att_genes if att == Attributes.CON]
        return sum(l)

    def vitality_points(self):
        l = [1 for att in self.att_genes if att == Attributes.VIT]
        return sum(l)