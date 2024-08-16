from src.sokoban import Sokoban,Direction, Symbol

def main():
    with open('inputs/input1', 'r') as file:
        level = [list(map(Symbol, line.strip('\n'))) for line in file]

    game = Sokoban(level)
    game.print_board()

    while not game.is_completed():
        move = input("Enter move (U/D/L/R): ").upper()
        direction_map = {
            'U': Direction.UP,
            'D': Direction.DOWN,
            'L': Direction.LEFT,
            'R': Direction.RIGHT
        }

        if move in direction_map:
            if game.move(direction_map[move]):
                game.print_board()
            else:
                print("Invalid move!")
        else:
            print("Invalid input!")

    print("Congratulations! You've completed the level!")

if __name__ == "__main__":
    main()
