from typing import List
from TP2.src.model.individual import Individual
from TP2.src.hyperparams.selection.abstract_selection import Selection


class CombinedSelection(Selection):
    def __init__(self, k: int, method_a: Selection, method_b: Selection, percentage_a: float):
        """
        :param k: Number of parents to select.
        :param method_a: First selection strategy.
        :param method_b: Second selection strategy.
        :param percentage_a: Percentage of individuals selected using method A (between 0 and 1).
        """
        super().__init__(k)
        self.method_a = method_a
        self.method_b = method_b
        self.percentage_a = percentage_a

    def select(self, population: List[Individual]) -> List[Individual]:
        # Number of individuals selected by method A
        k_a = int(self.k * self.percentage_a)
        # Number of individuals selected by method B
        k_b = self.k - k_a

        # Select k_a individuals using method A
        selected_a = self.method_a.select(population)[:k_a]

        # Select k_b individuals using method B
        selected_b = self.method_b.select(population)[:k_b]

        # Combine both selections
        return selected_a + selected_b
