import heapq
from typing import Callable

from TP1.src.node import Node
from TP1.src.search_methods.informed_abstract_method import InformedSearchMethod
from TP1.src.state import State


class GreedySearch(InformedSearchMethod):
    def search(self, initial_state: State):
        open_list = []
        heapq.heappush(open_list,
                       (Node(self.heuristic(initial_state), initial_state)))  # Usamos solo la heurística para la prioridad
        visited = set()
        while open_list:
            current_node = heapq.heappop(open_list)
            current_state = current_node.state
            self.reconstructed_path.append(current_state)
            if self.is_goal_state(current_state):
                return self.reconstruct_path(current_state)

            visited.add(current_state)

            for neighbor in self.get_neighbors(current_state):
                if neighbor not in visited:
                    visited.add(neighbor)
                    heapq.heappush(open_list, Node(self.heuristic(neighbor), neighbor))

        return None  # No se encontró solución

    # TODO: Implement
    def reconstruct_path(self, current_state):
        pass
