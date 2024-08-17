from collections import deque
from TP1.src.search_methods.abstract_method import SearchMethod
from TP1.src.sokoban import Direction
from TP1.src.state import State


class BFS(SearchMethod):
    def search(self, initial_state: State):
        queue = deque([initial_state])
        visited = set()
        visited.add(initial_state)

        while queue:
            current_state = queue.popleft()
            self.reconstructed_path.append(current_state)
            if self.is_goal_state(current_state):
                return self.reconstructed_path

            neighbors = self.get_neighbors(current_state)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return None

