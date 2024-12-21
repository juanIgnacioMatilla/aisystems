from typing import Callable

from search_methods.src.heuristics.manhattan_heuristic import manhattan_heuristic
from search_methods.src.sokoban import Sokoban
from search_methods.src.state import State

def weighted_manhattan_heuristic(targets: set, walls: set) -> Callable[[State], float]:
    def f(inner_state: State) -> float:
        total_penalty = 1

        for bx, by in inner_state.boxes:
            box_penalty = 0

            # 1. Proximity to Other Boxes
            neighbors = [
                (bx - 1, by), (bx + 1, by),  # Left and right
                (bx, by - 1), (bx, by + 1)  # Up and down
            ]
            adjacent_boxes = sum(1 for n in neighbors if n in inner_state.boxes)
            box_penalty += adjacent_boxes * 1.5  # Penalty for being adjacent to other boxes

            # 2. Distance from the Player
            player_distance = abs(inner_state.player_pos[0] - bx) + abs(inner_state.player_pos[1] - by)
            box_penalty += player_distance * 2  # Adding distance from player as a penalty

            # 3. Proximity to Walls
            if (bx - 1, by) in walls or (bx + 1, by) in walls:
                box_penalty += 1  # Penalty for being adjacent to a vertical wall (1 space away)
            if (bx - 2, by) in walls or (bx + 2, by) in walls:
                box_penalty += 0.5  # Additional penalty for being near a vertical wall (2 spaces away)

            if (bx, by - 1) in walls or (bx, by + 1) in walls:
                box_penalty += 1  # Penalty for being adjacent to a horizontal wall (1 space away)
            if (bx, by - 2) in walls or (bx, by + 2) in walls:
                box_penalty += 0.5  # Additional penalty for being near a horizontal wall (2 spaces away)

            # 4. Alignment with Goals
            min_distance_to_goal = min(abs(bx - gx) + abs(by - gy) for gx, gy in targets)
            if min_distance_to_goal > 0:  # If not aligned
                box_penalty += min_distance_to_goal * 3  # Increase weight for misalignment

            total_penalty += box_penalty

        return manhattan_heuristic(targets)(inner_state) + total_penalty
    return f