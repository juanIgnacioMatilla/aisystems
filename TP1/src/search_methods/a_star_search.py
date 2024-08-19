from queue import PriorityQueue

from TP1.src.node import Node
from TP1.src.search_methods.informed_abstract_method import InformedSearchMethod


class AStarSearch(InformedSearchMethod):
    def search(self):
        queue = PriorityQueue()
        queue.put(self.init_node)
        visited = set()
        visited.add(self.init_node.state)
        last_node = None
        while not queue.empty():
            current_node: Node = queue.get()
            if current_node.parent is not None:
                if current_node.parent not in self.node_dict_by_parent:
                    self.node_dict_by_parent[current_node.parent] = set()
                self.node_dict_by_parent[current_node.parent].add(current_node)
            if self.is_goal_state(current_node.state):
                self.frontier = queue.queue
                self.success = True
                return self.reconstruct_path(current_node)

            frontier = self.get_neighbours(current_node.state)
            for neighbour in frontier:
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.put(self.add_node(self.heuristic(current_node.state) + current_node.cost, neighbour, current_node))
            last_node = current_node
        return self.reconstruct_path(last_node)
