import heapq
from collections import deque
from queue import PriorityQueue

from search_methods.src.node import Node
from search_methods.src.search_methods.informed_abstract_method import InformedSearchMethod
from search_methods.src.state import State

class PrioritizedNodeTuple:
    def __init__(self, f_value : float, node : Node):
        self.f_value = f_value
        self.node = node

    def __lt__(self, other):
        return self.f_value < other.f_value

    def __eq__(self, other):
        return self.f_value == other.f_value

class GreedySearch(InformedSearchMethod):
    def search(self):
        frontier = PriorityQueue()
        frontier.put(PrioritizedNodeTuple(0, self.init_node))
        visited = set()
        visited.add(self.init_node.state)
        last_node = None
        while not frontier.empty():
            current_tuple: PrioritizedNodeTuple = frontier.get()
            self.explored_counter += 1
            if current_tuple.node.parent is not None:
                if current_tuple.node.parent not in self.node_dict_by_parent:
                    self.node_dict_by_parent[current_tuple.node.parent] = set()
                self.node_dict_by_parent[current_tuple.node.parent].add(current_tuple.node)
            if self.is_goal_state(current_tuple.node.state):
                self.frontier = [prioritized_tuple.node for prioritized_tuple in frontier.queue]
                self.success = True
                return self.reconstruct_path(current_tuple.node)

            successors = self.get_successors(current_tuple.node.state)
            for successor in successors:
                if successor not in visited:
                    visited.add(successor)
                    h_value: float  = self.heuristic(successor)
                    node = self.add_node(current_tuple.node.g_value + 1, successor, current_tuple.node, h_value)
                    frontier.put(PrioritizedNodeTuple(node.h_value,node))
            last_node = current_tuple.node
        return self.reconstruct_path(last_node)
