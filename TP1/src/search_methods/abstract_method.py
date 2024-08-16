from abc import ABC, abstractmethod
from TP1.src.sokoban import Sokoban
class SearchMethod(ABC):

    def __init__(self, sokoban: Sokoban):
        self.sokoban = sokoban
        self.reconstructed_path = []
    @abstractmethod
    def search(self, initial_state):
        pass