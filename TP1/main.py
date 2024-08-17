from src.sokoban import Sokoban, Symbol
from src.state import State

from TP1.src.gif_generator import create_gif, delete_frames, generate_gif, print_path
from TP1.src.search_methods.bfs import BFS
from TP1.src.search_methods.dfs import DFS


def main():
    with open("inputs/input3", "r") as file:
        level = [list(map(Symbol, line.strip("\n"))) for line in file]

    game = Sokoban(level)
    game.print_board()

    # search method BFS
    # print("BFS")
    # bfs = BFS(game)
    # path = bfs.search(State(game.player_pos, game.boxes))
    print("DFS")
    dfs = DFS(game)
    path = dfs.search(State(game.player_pos, game.boxes))
    if path:
        print("Search path: ", path)
        print("Length: ", len(path))
        print()
        # Visualize the path
        generate_gif(path, game, gif_name="input3_solution.gif")
    else:
        print("No solution found.")


if __name__ == "__main__":
    main()
