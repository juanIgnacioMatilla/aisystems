import sys
import json

import numpy as np

from TP1.src.search_methods.a_star_search_optimized import AStarOptimizedSearch
from TP1.src.search_methods.bfs_optimized import BFSOptimized
from TP1.src.search_methods.dfs_optimized import DFSOptimized
from TP1.utils.heuristic_builder import HeuristicBuilder
from TP1.utils.plot_results import plot_scatter
from TP1.utils.print_results import print_results
from TP1.src.search_methods.a_star_search import AStarSearch
from TP1.src.search_methods.dfs import DFS
from src.sokoban import Sokoban, Symbol
from TP1.src.search_methods.bfs import BFS
from TP1.src.search_methods.greedy_search import GreedySearch

informed_methods = ['GGS', 'A*', 'A*_Optimized']
methods_dict = {'GGS': GreedySearch, 'A*': AStarSearch, 'A*_Optimized': AStarOptimizedSearch, "BFS": BFS, "DFS": DFS,
                "BFS_Optimized": BFSOptimized, "DFS_Optimized": DFSOptimized}


def main():
    with open('TP1/config.json', 'r') as file:
        config = json.load(file)

        runs = config['runs_per_method'] if 'runs_per_method' in config else 1
        maps = config['maps']
        for soko_map in maps:
            print(f"Running map {soko_map}")
            with open("TP1/inputs/" + soko_map, "r") as map_file:
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
                result['time_error'] = std_dev_time / np.sqrt(runs)

                if heuristic and not secondary_heuristic:
                    result["method"] = f"{result['method']} ({heuristic})"
                elif heuristic and secondary_heuristic:
                    result["method"] = f"{result['method']} ({heuristic}) ({secondary_heuristic})"

                results_list.append(result)

            #split the soko_map name to get the map name after the / character
            soko_map = soko_map.split("/")[-1]
            # Graficar los resultados
            plot_scatter(results_list, runs, soko_map)


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


if __name__ == "__main__":
    main()
