import copy
import sys
import json

import numpy as np

from search_methods.src.search_methods.a_star_search_optimized import AStarOptimizedSearch
from search_methods.src.search_methods.bfs_optimized import BFSOptimized
from search_methods.src.search_methods.dfs_optimized import DFSOptimized
from search_methods.utils.gif_generator import generate_gif
from search_methods.utils.heuristic_builder import HeuristicBuilder
from search_methods.utils.plot_results import plot_scatter
from search_methods.utils.print_results import print_results
from search_methods.src.search_methods.a_star_search import AStarSearch
from search_methods.src.search_methods.dfs import DFS
from src.sokoban import Sokoban, Symbol
from search_methods.src.search_methods.bfs import BFS
from search_methods.src.search_methods.greedy_search import GreedySearch

informed_methods = ['GGS', 'A*', 'A*_Optimized']
methods_dict = {'GGS': GreedySearch, 'A*': AStarSearch, 'A*_Optimized': AStarOptimizedSearch, "BFS": BFS, "DFS": DFS,
                "BFS_Optimized": BFSOptimized, "DFS_Optimized": DFSOptimized}



def main():
    with open('search_methods/config.json', 'r') as file:
        config = json.load(file)

        runs = config['runs_per_method'] if 'runs_per_method' in config else 1
        maps = config['maps']
        for soko_map in maps:
            print(f"Running map {soko_map}")
            with open("search_methods/inputs/" + soko_map, "r") as map_file:
                level = [list(map(Symbol, line.strip("\n"))) for line in map_file]

            # Game initialization
            game = Sokoban(level)
            game.print_board()
            heuristic_builder = HeuristicBuilder(game)

            results_list = []
            for search_method in config['search_methods']:

                (method, heuristic, secondary_heuristic, weight, combination1, combination2) = get_method(search_method,
                                                                                                          heuristic_builder.heuristic_dict.keys())

                print(
                    f"Running method {method} {heuristic if heuristic else ''} {secondary_heuristic if secondary_heuristic else ''} {weight if weight else ''}")

                # Ejecutar el métod de búsqueda y guardar los resultados
                m = methods_dict[method]
                h = heuristic_builder.get_heuristic(heuristic, secondary_heuristic, weight, combination1, combination2)
                times = []
                for i in range(runs):
                    result = print_results(game, m, h)
                    times.append(result['time'])

                # Calcular la media y el error (desviación estándar)
                mean_time = np.mean(times)
                std_dev_time = np.std(times)

                # Añadir la media y el error al resultado final
                result['time'] = mean_time
                result['time_error'] = std_dev_time
                if heuristic and not secondary_heuristic:
                    result["method"] = f"{result['method']} ({heuristic})"
                elif heuristic and secondary_heuristic:
                    result["method"] = f"{result['method']} ({heuristic}) ({secondary_heuristic})"

                # Printing the results according to the structure shown in the image
                print(f" - Result: {'Success' if result['success'] else 'Failure'}")
                print(f" - Solution cost: {result['path_length']}")
                print(f" - Number of expanded nodes: {result['expanded_nodes']}")
                print(f" - Number of frontier nodes: {result['frontier_nodes']}")
                print(f" - Processing time in seconds: {result['time']:.6f} +- {result['time_error']:.6f}")
                results_list.append(result)
                actions = []
                for index in range(1, len(result['path'])):
                    prev_node = result['path'][index - 1]
                    current_node = result['path'][index]
                    movement = get_movement(prev_node.state.player_pos, current_node.state.player_pos)
                    actions.append(movement)
                print(" - Actions: ", "->".join(actions))
                if 'generate_gif' in config and config['generate_gif'] is True:
                    print("Generating GIF...")
                    copy_game = copy.deepcopy(game)
                    generate_gif(result['path'], copy_game, f'{soko_map.split("/")[-1]}_{result["method"]}.gif')
            # #split the soko_map name to get the map name after the / character
            # soko_map = soko_map.split("/")[-1]
            # # Graficar los resultados
            # plot_scatter(results_list, runs, soko_map)


def get_method(search_method, heuristics_list):
    heuristic = None
    secondary_heuristic = None
    weight = None
    combination1 = None
    combination2 = None
    if "method" not in search_method:
        print("method not found")
        sys.exit(1)
    method = search_method["method"]

    if method in informed_methods:
        if "heuristic" not in search_method:
            print("heuristic not found")
            sys.exit(1)
        if search_method["heuristic"] in heuristics_list:
            heuristic = search_method["heuristic"]
        else:
            print("heuristic does not exist")
            sys.exit(1)
        if "secondary_heuristic" in search_method:
            if search_method["secondary_heuristic"] in heuristics_list:
                secondary_heuristic = search_method["secondary_heuristic"]
        if "weight" in search_method:
            weight = search_method["weight"]
        if "combined1" in search_method:
            if search_method["combined1"] in heuristics_list:
                combination1 = search_method["combined1"]
        if "combined2" in search_method:
            if search_method["combined2"] in heuristics_list:
                combination2 = search_method["combined2"]
            else:
                print("heuristic does not exist")
                sys.exit(1)
    elif methods_dict.keys().__contains__(method):
        return method, heuristic, secondary_heuristic, weight, combination1, combination2
    else:
        print("method does not exist")
    return method, heuristic, secondary_heuristic, weight, combination1, combination2


def get_movement(prev_position, current_position):
    row_diff = current_position[0] - prev_position[0]
    col_diff = current_position[1] - prev_position[1]

    if row_diff == -1:
        return "Up"
    elif row_diff == 1:
        return "Down"
    elif col_diff == -1:
        return "Left"
    elif col_diff == 1:
        return "Right"
    else:
        return "No movement"


if __name__ == "__main__":
    main()
