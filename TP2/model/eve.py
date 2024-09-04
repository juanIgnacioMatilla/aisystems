import math

from TP2.model.individual import Individual


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
        return 0.5 - (3 * height - 5)**4 / ((3 * height - 5)**2 + height / 2)

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
    def calculate_fitness(individual: Individual):
        attack = EVE.calculate_attack(
            individual.agility_total,
            individual.intelligence_total,
            individual.force_total,
            individual.atm
        )
        defense = EVE.calculate_defense(
            individual.vigor_total,
            individual.intelligence_total,
            individual.constitution_total,
            individual.dem
        )
        performances = {
            'Warrior': 0.6 * attack + 0.4 * defense,
            'Archer': 0.9 * attack + 0.1 * defense,
            'Guardian': 0.1 * attack + 0.9 * defense,
            'Mage': 0.8 * attack + 0.3 * defense
        }
        return performances.get(individual.type, 0)
