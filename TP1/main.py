import cProfile
import time
import timeit

from TP1.src.heuristics.blocked_heuristic import blocked_heuristic
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


def main():
    with open("TP1/inputs/input4", "r") as file:
        level = [list(map(Symbol, line.strip("\n"))) for line in file]

    game = Sokoban(level)
    game.print_board()

    print_results(game, BFS)

    print_results(game, DFS)

    print_results(game, AStarSearch, trivial_heuristic)


if __name__ == "__main__":
    main()
