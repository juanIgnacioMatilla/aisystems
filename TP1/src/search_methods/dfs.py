from TP1.src.search_methods.abstract_method import SearchMethod
from TP1.src.state import State


class DFS(SearchMethod):
    def search(self):
        stack = [self.init_node]
        visited = set()
        visited.add(self.init_node.state)
        last_node = None
        while stack:
            current_node = stack.pop()
            if current_node.parent is not None:
                if current_node.parent not in self.node_dict_by_parent:
                    self.node_dict_by_parent[current_node.parent] = set()
                self.node_dict_by_parent[current_node.parent].add(current_node)
            if self.is_goal_state(current_node.state):
                return self.reconstruct_path(current_node)

            neighbours = self.get_neighbours(current_node.state)
            for neighbour in neighbours:
                if neighbour not in visited:
                    visited.add(neighbour)
                    stack.append(self.add_node(0, neighbour, current_node))
            last_node = current_node
        return self.reconstruct_path(last_node)  # error exit
