from typing import Optional

from TP1.src.state import State


class Node:
    def __init__(
        self, g_value: float, id: int, state: State,  parent: Optional["Node"] = None, h_value: Optional[float] = None
    ):
        self.g_value = g_value
        self.h_value = h_value
        self.state = state
        self.parent = parent
        self.id = id

    def __hash__(self):
        return hash((self.id, self.state))

    def __lt__(self, other):
        return self.g_value < other.g_value

    def __le__(self, other):
        return self.g_value <= other.g_value

    def __gt__(self, other):
        return self.g_value > other.g_value

    def __ge__(self, other):
        return self.g_value >= other.g_value

    def __eq__(self, other):
        return self.g_value == other.g_value and self.state == other.state

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return f"Node(g_value={self.g_value}, h_value={self.h_value}, state={self.state})"
