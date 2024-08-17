from TP1.src.search_methods.greedy_search import GreedySearch
from src.sokoban import Sokoban, Direction, Symbol
from src.state import State
from src.search_methods.bfs import BFS
from src.search_methods.dfs import DFS
from src.heuristics.trivial_heuristic import trivial_heuristic


def main():
    with open('inputs/input3', 'r') as file:
        level = [list(map(Symbol, line.strip('\n'))) for line in file]

    game = Sokoban(level)
    game.print_board()

    # search method BFS
    print("BFS")
    bfs = BFS(game)
    bfs.search(State(game.player_pos, game.boxes))
    print("Length: ", len(bfs.reconstructed_path))
    print()
    # search method DFS
    print("DFS")
    dfs = DFS(game)
    dfs.search(State(game.player_pos, game.boxes))
    print("Length: ", len(dfs.reconstructed_path))
    print()
    # search method Greedy Search
    print("Greedy")
    greedy = GreedySearch(game, trivial_heuristic)
    greedy.search(State(game.player_pos, game.boxes))
    print("Length: ", len(greedy.reconstructed_path))

# # use arrows for movement
#     while not game.is_completed():
#         move = input("Enter your move: ")
#         direction_map = {
#             'w': Direction.UP,
#             's': Direction.DOWN,
#             'a': Direction.LEFT,
#             'd': Direction.RIGHT
#         }
#
#         if move in direction_map:
#             if game.move(direction_map[move]):
#                 game.print_board()
#             else:
#                 print("Invalid move!")
#         else:
#             print("Invalid input!")
#
#     print("Congratulations! You've completed the level!")

if __name__ == "__main__":
    main()
