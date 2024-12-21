from abc import abstractmethod
from typing import Callable

from search_methods.src.node import Node
from search_methods.src.search_methods.abstract_method import SearchMethod
from search_methods.src.sokoban import Sokoban
from search_methods.src.state import State


class InformedSearchMethod(SearchMethod):
    def __init__(self, targets: set, walls: set, initial_state: State, heuristic: Callable[[State], float]):
        super().__init__(targets, walls, initial_state)
        self.heuristic = heuristic

    @abstractmethod
    def search(self) -> list[Node]:
        pass
