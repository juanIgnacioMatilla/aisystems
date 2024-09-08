import math

from TP2.src.model.chromosome import Chromosome
from TP2.src.model.individual_types import IndividualTypes


class EVE:
    @staticmethod
    def calculate_force_total(strength_points):
        return 100 * math.tanh(0.01 * strength_points)

    @staticmethod
    def calculate_agility_total(agility_points):
        return math.tanh(0.01 * agility_points)

    @staticmethod
    def calculate_intelligence_total(intelligence_points):
        return 0.6 * math.tanh(0.01 * intelligence_points)

    @staticmethod
    def calculate_vigor_total(vigor_points):
        return math.tanh(0.01 * vigor_points)

    @staticmethod
    def calculate_constitution_total(constitution_points):
        return 100 * math.tanh(0.01 * constitution_points)

    @staticmethod
    def calculate_atm(height):
        return 0.5 - (3 * height - 5)**4 + ((3 * height - 5)**2) + height / 2

    @staticmethod
    def calculate_dem(height):
        return 2 + (3 * height - 5)**4 - (3 * height - 5)**2 - height / 2

    @staticmethod
    def calculate_attack(agility_total, intelligence_total, force_total, atm):
        return (agility_total + intelligence_total) * force_total * atm

    @staticmethod
    def calculate_defense(vigor_total, intelligence_total, constitution_total, dem):
        return (vigor_total + intelligence_total) * constitution_total * dem

    @staticmethod
    def calculate_fitness(individual_type: IndividualTypes, chromosome: Chromosome):
        height = chromosome.height
        # Calculate attributes using EVE class
        force_total = EVE.calculate_force_total(chromosome.strength_points)
        agility_total = EVE.calculate_agility_total(chromosome.agility_points)
        intelligence_total = EVE.calculate_intelligence_total(chromosome.intelligence_points)
        vigor_total = EVE.calculate_vigor_total(chromosome.vigor_points)
        constitution_total = EVE.calculate_constitution_total(chromosome.constitution_points)
        atm = EVE.calculate_atm(height)
        dem = EVE.calculate_dem(height)

        attack = EVE.calculate_attack(
            agility_total,
            intelligence_total,
            force_total,
            atm
        )
        defense = EVE.calculate_defense(
            vigor_total,
            intelligence_total,
            constitution_total,
            dem
        )
        performances = {
            IndividualTypes.WARRIOR: 0.6 * attack + 0.4 * defense,
            IndividualTypes.ARCHER: 0.9 * attack + 0.1 * defense,
            IndividualTypes.GUARDIAN: 0.1 * attack + 0.9 * defense,
            IndividualTypes.MAGE: 0.8 * attack + 0.3 * defense
        }
        return performances.get(individual_type, 0)
