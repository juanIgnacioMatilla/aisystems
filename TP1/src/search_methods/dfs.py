from TP1.src.node import Node
from TP1.src.search_methods.abstract_method import SearchMethod
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

            neighbours = self.get_neighbours(current_state)
            for neighbour in neighbours:
                if neighbour not in visited:
                    self.reconstructed_path.append(Node(1, neighbour))
                    visited.add(neighbour)
                    stack.append(neighbour)
        return self.reconstructed_path  #
