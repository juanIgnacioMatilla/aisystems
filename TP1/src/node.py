from TP1.src.state import State


class Node:
    def __init__(self, cost: float, state: State):
        self.cost = cost
        self.state = state

    def __lt__(self, other):
        return self.cost < other.cost

    def __le__(self, other):
        return self.cost <= other.cost

    def __gt__(self, other):
        return self.cost > other.cost

    def __ge__(self, other):
        return self.cost >= other.cost

    def __eq__(self, other):
        return self.cost == other.cost and self.state == other.state

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return f"Node(cost={self.cost}, state={self.state})"
