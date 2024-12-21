from scipy.optimize import linear_sum_assignment
import numpy as np
from typing import Callable
from search_methods.src.state import State

def minimum_matching_distance_heuristic(targets: set) -> Callable[[State], float]:
    def f(inner_state: State) -> float:
        # Convert boxes and targets to lists for indexing
        boxes = list(inner_state.boxes)
        targets_list = list(targets)

        # Create a cost matrix where cost_matrix[i][j] is the Manhattan distance from box i to target j
        cost_matrix = np.zeros((len(boxes), len(targets_list)))
        for i, (bx, by) in enumerate(boxes):
            for j, (tx, ty) in enumerate(targets_list):
                cost_matrix[i, j] = abs(bx - tx) + abs(by - ty)

        # Use the Hungarian algorithm to find the optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # The total minimum matching distance
        return cost_matrix[row_ind, col_ind].sum()

    return f
