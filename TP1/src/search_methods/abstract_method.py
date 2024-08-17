from abc import ABC, abstractmethod
from collections import deque

from TP1.src.node import Node
from TP1.src.sokoban import Direction, Sokoban
from TP1.src.state import State


class SearchMethod(ABC):

    def __init__(self, sokoban: Sokoban):
        self.sokoban = sokoban
        self.node_dict_by_parent: dict[Node, set[Node]] = {}
        self.node_counter: int = 0
        self.init_node: Node | None = None

        # Search tiene que crear el node inicial y setearselo a la funcion porq sino explota todo

    @abstractmethod
    def search(self, initial_state: State) -> list[Node]:
        pass

    def is_goal_state(self, state: State):
        if state.is_completed(self.sokoban.targets):
            print(state)
            return True
        return False

    def get_neighbours(self, state: State) -> list[State]:
        neighbours = []

        for direction in Direction:
            if state.can_move(direction, self.sokoban.walls):
                new_state = state.copy_move(direction, self.sokoban.walls)
                neighbours.append(new_state)
        return neighbours

    def reconstruct_path(self, node: Node | None) -> list[Node]:
        if node is None:
            return []
        parent_node = node.parent
        node_list = deque([node])
        while parent_node is not None:
            node_list.appendleft(parent_node)
            parent_node = parent_node.parent

        return list(node_list)

    def add_node(self, cost: float, state: State, parent: Node | None) -> Node:
        node = Node(cost, self.node_counter, state, parent)
        self.node_counter += 1
        return node