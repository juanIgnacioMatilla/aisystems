import time
from typing import Callable

from TP1.src.search_methods.abstract_method import SearchMethod
from TP1.src.sokoban import Sokoban
from TP1.src.state import State


def print_results(game: Sokoban, search_method_class: type, heuristic: Callable[[State], float] = None):
    initial_state = State(game.player_pos, game.boxes)
    print(search_method_class.__name__)
    if heuristic is None:
        search_method = search_method_class(game.targets, game.walls, initial_state)
    else:
        search_method = search_method_class(game.targets, game.walls, initial_state, heuristic)
    start_time = time.time()
    path = search_method.search()
    end_time = time.time()
    print("Success: ", search_method.success)
    print("Expanded nodes count: ", search_method.explored_counter)
    print("Total frontier nodes count: ", search_method.node_counter)
    print("Frontier nodes left count: ", len(search_method.frontier))
    print("Solution path: ", path)
    print("Time: ", end_time - start_time)
    print("Length: ", len(path))


    print()
    # Visualize the path
    # generate_gif(path, game, gif_name="sokoban_solution.gif")
