from abc import ABC, abstractmethod
from TP1.src.sokoban import Sokoban, Direction
from TP1.src.state import State


class SearchMethod(ABC):

    def __init__(self, sokoban: Sokoban):
        self.sokoban = sokoban
        self.reconstructed_path = []
    @abstractmethod
    def search(self, initial_state):
        pass

    def is_goal_state(self, state):
        if state.is_completed(self.sokoban.targets):
            print(state)
            return True
        return False

    def get_neighbors(self, state: State):
        neighbors = []

        for direction in Direction:
            if state.can_move(direction, self.sokoban.walls):
                new_state = state.copy_move(direction, self.sokoban.walls)
                neighbors.append(new_state)

        return neighbors