from TP1.src.heuristics.blocked_heuristic import blocked_heuristic
from TP1.src.heuristics.trivial_heuristic import trivial_heuristic
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

    initial_state = State(game.player_pos, game.boxes)
    print("BFS")
    bfs = BFS(game.targets, game.walls, initial_state)
    path = bfs.search()
    if path:
        print("Search path: ", path)
        print("Length: ", len(path))
        print("Node counter: ",bfs.node_counter)
        print()
        # Visualize the path
        # generate_gif(path, game, gif_name="input4_solution.gif")
    else:
        print("No solution found.")

    print("Greedy")
    greedy = GreedySearch(
        game.targets,
        game.walls,
        initial_state,
        # trivial_heuristic
        blocked_heuristic(initial_state, game.walls, manhattan_heuristic(initial_state, game.targets))
    )
    path = greedy.search()
    if path:
        print("Search path: ", path)
        print("Length: ", len(path))
        print("Node counter: ",greedy.node_counter)
        print()
        # Visualize the path
        generate_gif(path, game, gif_name="input4_solution.gif")
    else:
        print("No solution found.")


if __name__ == "__main__":
    main()
