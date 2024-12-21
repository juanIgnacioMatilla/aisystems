import random
from typing import NamedTuple, List

from genetic_algorithms.src.model.attributes import Attributes


class Chromosome(NamedTuple):
    height: float
    att_genes: List[Attributes]

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