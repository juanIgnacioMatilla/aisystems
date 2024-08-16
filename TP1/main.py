from src.sokoban import Sokoban, Direction, Symbol
from src.state import State
def main():
    with open('inputs/input5', 'r') as file:
        level = [list(map(Symbol, line.strip('\n'))) for line in file]

    game = Sokoban(level)
    game.print_board()

    # search method BFS
    from src.search_methods.bfs import BFS
    bfs = BFS(game)
    bfs.search(State(game.player_pos,game.boxes))
    print("BFS")
    # print("Path: ", bfs.reconstructed_path)
    print("Length: ", len(bfs.reconstructed_path))


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
