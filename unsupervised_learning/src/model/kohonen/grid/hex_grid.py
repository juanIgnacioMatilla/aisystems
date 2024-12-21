from typing import Tuple, List
from TP4.src.model.kohonen.grid.grid import Grid
from TP4.src.model.kohonen.neuron import Neuron


class HexGrid(Grid):

    def get_neighbors(self, coords: Tuple[int, int], r) -> List[Neuron]:
        row, col = coords
        neighbors = []
        # Iterate through each radius level
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                if abs(dr + dc) <= r:  # Check if within hexagonal radius
                    new_row = row + dr
                    new_col = col + dc

                    # Adjust column based on whether the row is even or odd
                    if dr % 2 == 0:  # Even row
                        new_col += 0
                    else:  # Odd row
                        new_col += dc % 2  # Add 1 if dc is odd

                    # Check if the new position is within bounds
                    if 0 <= new_row < self.size and 0 <= new_col < self.size:
                        # Exclude the central element
                        if (new_row, new_col) != (row, col):
                            neighbors.append(self.matrix[new_row, new_col])
        return neighbors
