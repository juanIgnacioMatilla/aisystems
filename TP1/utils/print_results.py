import time
from typing import Callable
from TP1.src.sokoban import Sokoban
from TP1.src.state import State
from TP1.utils.gif_generator import generate_gif


def print_results(game: Sokoban, search_method_class: type, heuristic: Callable[[State], float] = None):
    initial_state = State(game.player_pos, game.boxes)
    if heuristic is None:
        search_method = search_method_class(game.targets, game.walls, initial_state)
    else:
        search_method = search_method_class(game.targets, game.walls, initial_state, heuristic)

    start_time = time.time()
    path = search_method.search()
    end_time = time.time()
    results = {
        "method": search_method_class.__name__,
        "heuristic": heuristic.__name__ if heuristic else "None",
        "success": search_method.success,
        "expanded_nodes": search_method.explored_counter,
        "total_nodes": search_method.node_counter,
        "frontier_nodes": len(search_method.frontier),
        "time": end_time - start_time,
        "path_length": len(path) - 1,
        "path": path
    }

    return results
