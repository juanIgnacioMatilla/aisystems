from collections import deque

from TP1.src.node import Node
from TP1.src.search_methods.abstract_method import SearchMethod
from TP1.src.state import State


class BFS(SearchMethod):
    def search(self, initial_state: State):
        self.init_node = self.add_node(0, initial_state, None)
        queue = deque([self.init_node])
        visited = set()
        visited.add(initial_state)
        last_node = None
        while queue:
            current_node: Node = queue.popleft()
            self.node_dict_by_parent[current_node.parent].add(current_node)
            if self.is_goal_state(current_node.state):
                return self.reconstruct_path(current_node)

            neighbours = self.get_neighbours(current_node.state)
            for neighbour in neighbours:
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(self.add_node(0, neighbour, current_node))
            last_node = current_node
        return self.reconstruct_path(last_node)
