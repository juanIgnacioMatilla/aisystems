import heapq

from TP1.src.search_methods.informed_abstract_method import InformedSearchMethod
from TP1.src.state import State


class GreedySearch(InformedSearchMethod):
    def search(self, initial_state: State):
        open_list = []
        self.init_node = self.add_node(
            self.heuristic(initial_state), initial_state, None
        )
        last_node = None
        heapq.heappush(
            open_list, self.init_node
        )  # Usamos solo la heurística para la prioridad
        visited = set()
        while open_list:
            current_node = heapq.heappop(open_list)

            if current_node.parent is not None:
                if current_node.parent not in self.node_dict_by_parent:
                    self.node_dict_by_parent[current_node.parent] = set()
                self.node_dict_by_parent[current_node.parent].add(current_node)

            if self.is_goal_state(current_node.state):
                return self.reconstruct_path(current_node)

            visited.add(current_node.state)

            for neighbour in self.get_neighbours(current_node.state):
                if neighbour not in visited:
                    visited.add(neighbour)
                    heapq.heappush(
                        open_list,
                        self.add_node(
                            self.heuristic(neighbour), neighbour, current_node
                        ),
                    )
            last_node = current_node

        return self.reconstruct_path(last_node)  # No se encontró solución
