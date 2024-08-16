from TP1.src.search_methods.abstract_method import SearchMethod
from TP1.src.sokoban import Direction
from TP1.src.state import State

class DFS(SearchMethod):
    def search(self, initial_state: State):
        stack = [initial_state]
        visited = set()
        visited.add(initial_state)

        while stack:
            current_state = stack.pop()
            if self.is_goal_state(current_state):
                return self.reconstructed_path

            neighbors = self.get_neighbors(current_state)
            for neighbor in neighbors:
                if neighbor not in visited:
                    self.reconstructed_path.append(neighbor)
                    visited.add(neighbor)
                    stack.append(neighbor)

        return None  #

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
