import cProfile
import sys
import time
import timeit
import json
from typing import Callable

from sympy.series.limits import heuristics
from sympy.strategies.core import switch

from TP1.src.heuristics.blocked_heuristic import blocked_heuristic
from TP1.src.heuristics.heuristic_builder import HeuristicBuilder
from TP1.src.heuristics.trivial_heuristic import trivial_heuristic
from TP1.src.print_results import print_results
from TP1.src.search_methods.a_star_search import AStarSearch
from TP1.src.search_methods.dfs import DFS
from src.sokoban import Sokoban, Symbol
from src.state import State

from TP1.src.gif_generator import create_gif, delete_frames, generate_gif, print_path
from TP1.src.search_methods.bfs import BFS
from TP1.src.search_methods.greedy_search import GreedySearch
from TP1.src.heuristics.manhattan_heuristic import manhattan_heuristic


informed_methods = ['GGS', 'A*']
methods_dict = {'GGS': GreedySearch, 'A*': AStarSearch, "BFS": BFS, "DFS": DFS}
def main():
    with open("TP1/inputs/input4", "r") as file:
        level = [list(map(Symbol, line.strip("\n"))) for line in file]


    game = Sokoban(level)
    game.print_board()
    heuristic_builder = HeuristicBuilder(game)
    with open('TP1/config.json', 'r') as file:
        config = json.load(file)
        if 'runs_per_method' not in config:
            print("runs per method not specified in config file:")
            sys.exit(1)
        runs = config['runs_per_method'] if 'runs_per_method' in config else 1
        for search_method in config['search_methods']:
            (method, heuristic, secondary_heuristic) = get_method(search_method, heuristic_builder.available_heuristics)
            print(method, heuristic, secondary_heuristic)
            m = methods_dict[method]
            h = heuristic_builder.get_heuristic(heuristic, secondary_heuristic)
            for i in range(runs):
                print_results(game, m, h)


    #print_results(game, BFS)

    #print_results(game, DFS)

    #print_results(game, AStarSearch,)







def get_method(search_method, heuristics_list):
    heuristic = None
    secondary_heuristic = None
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
            else:
                print("heuristic does not exist")
                sys.exit(1)
    elif methods_dict.keys().__contains__(method):
        return method, heuristic, secondary_heuristic
    else:
        print("method does not exist")
    return method, heuristic, secondary_heuristic





if __name__ == "__main__":
    main()
