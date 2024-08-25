from TP1.src.search_methods.abstract_method import SearchMethod
from TP1.src.state import State


class DFS(SearchMethod):
    def search(self):
        frontier = [self.init_node]
        visited = set()
        visited.add(self.init_node.state)
        last_node = None
        while frontier:
            current_node = frontier.pop()
            self.explored_counter += 1
            if current_node.parent is not None:
                if current_node.parent not in self.node_dict_by_parent:
                    self.node_dict_by_parent[current_node.parent] = set()
                self.node_dict_by_parent[current_node.parent].add(current_node)
            if self.is_goal_state(current_node.state):
                self.frontier = frontier
                self.success = True
                return self.reconstruct_path(current_node)

            successors = self.get_successors(current_node.state)
            for successor in successors:
                if successor not in visited:
                    visited.add(successor)
                    frontier.append(self.add_node(1, successor, current_node))
            last_node = current_node
        return self.reconstruct_path(last_node)  # error exit
