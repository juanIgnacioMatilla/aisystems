from search_methods.src.search_methods.abstract_method import SearchMethod
from search_methods.src.sokoban import Direction
from search_methods.src.state import State

class IDDFS(SearchMethod):
    def search(self,initial_state: State):
        depth = 0

        while True:
            # Realizamos una búsqueda en profundidad con un límite de profundidad `depth`
            result = self.depth_limited_search(initial_state, depth)
            if result is not None:
                return result  # Se encontró una solución
            depth += 1  # Incrementamos la profundidad límite para la siguiente iteración

    def depth_limited_search(self, state: State, limit: int):
        return self.dls_recursive(state, limit, set())

    def dls_recursive(self, state: State, limit: int, visited):
        if self.is_goal_state(state):
            return self.reconstructed_path

        if limit == 0:
            return None  # Al alcanzar el límite, no seguimos explorando más

        visited.add(state)
        successors = self.get_successors(state)

        for successor in successors:
            if successor not in visited:
                self.reconstructed_path.append(successor)
                result = self.dls_recursive(successor, limit - 1, visited)
                if result is not None:
                    return result
                self.reconstructed_path.pop()  # Retroceder si no es solución válida

        visited.remove(state)
        return None

    def is_goal_state(self, state):
        if state.is_completed(self.sokoban.targets):
            print(state)
            return True
        return False

    def get_successors(self, state: State):
        successors = []

        for direction in Direction:
            if state.can_move(direction, self.sokoban.walls):
                new_state = state.copy_move(direction, self.sokoban.walls)
                successors.append(new_state)

        return successors
