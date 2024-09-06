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

    # Ensure total points match the target_total pints
    total_points = sum([
        chromosome.strength_points,
        chromosome.agility_points,
        chromosome.intelligence_points,
        chromosome.vigor_points,
        chromosome.constitution_points
    ])
    # If total points do not match the target, redistribute them
    if total_points != target_total:
        points = distribute_points(target_total, 5)
        strength, agility, intelligence, vigor, constitution = points
        print(f"NORMALIZING")
    else:
        strength = chromosome.strength_points
        agility = chromosome.agility_points
        intelligence = chromosome.intelligence_points
        vigor = chromosome.vigor_points
        constitution = chromosome.constitution_points

    # Return the normalized chromosome
    return Chromosome(
        height=height,
        strength_points=strength,
        agility_points=agility,
        intelligence_points=intelligence,
        vigor_points=vigor,
        constitution_points=constitution
    )


def distribute_points(total_points: int, num_attributes: int) -> List[int]:
    """Distributes total_points randomly across num_attributes attributes."""
    points = [0] * num_attributes
    for _ in range(total_points):
        # Randomly assign a point to one of the attributes
        points[random.randint(0, num_attributes - 1)] += 1
    return points
