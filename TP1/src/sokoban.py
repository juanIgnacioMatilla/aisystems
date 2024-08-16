from enum import Enum

class Symbol(Enum):
    WALL = '#'
    FREE = ' '
    BOX = '$'
    PLAYER = '@'
    TARGET = '.'
    BOX_ON_TARGET = '*'
    PLAYER_ON_TARGET = '+'

class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

class Sokoban:
    def __init__(self, level):
        self.level = level
        self.player_pos = None
        self.boxes = set()
        self.targets = set()
        self.walls = set()
        self.parse_level()

    def parse_level(self):
        for y, row in enumerate(self.level):
            for x, cell in enumerate(row):
                if cell == Symbol.PLAYER:
                    self.player_pos = (x, y)
                elif cell == Symbol.BOX:
                    self.boxes.add((x, y))
                elif cell == Symbol.TARGET:
                    self.targets.add((x, y))
                elif cell == Symbol.WALL:
                    self.walls.add((x, y))
                elif cell == Symbol.BOX_ON_TARGET:
                    self.boxes.add((x, y))
                    self.targets.add((x, y))
                elif cell == Symbol.PLAYER_ON_TARGET:
                    self.player_pos = (x, y)
                    self.targets.add((x, y))

    def move(self, direction: Direction):
        dx, dy = direction.value
        new_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)

        if new_pos in self.walls:
            return False

        if new_pos in self.boxes:
            new_box_pos = (new_pos[0] + dx, new_pos[1] + dy)
            if new_box_pos in self.walls or new_box_pos in self.boxes:
                return False
            self.boxes.remove(new_pos)
            self.boxes.add(new_box_pos)

        self.player_pos = new_pos
        return True

    def is_completed(self):
        return self.boxes == self.targets

    def print_board(self):
        for y in range(len(self.level)):
            for x in range(len(self.level[y])):
                if (x, y) == self.player_pos:
                    if (x, y) in self.targets:
                        print(Symbol.PLAYER_ON_TARGET.value, end='')
                    else:
                        print(Symbol.PLAYER.value, end='')
                elif (x, y) in self.boxes:
                    if (x, y) in self.targets:
                        print(Symbol.BOX_ON_TARGET.value, end='')
                    else:
                        print(Symbol.BOX.value, end='')
                elif (x, y) in self.targets:
                    print(Symbol.TARGET.value, end='')
                elif (x, y) in self.walls:
                    print(Symbol.WALL.value, end='')
                else:
                    print(Symbol.FREE.value, end='')
            print()

    def can_move(self, direction: Direction):
        dx, dy = direction.value
        new_player_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)

        if new_player_pos in self.walls:
            return False

        new_boxes = set(self.boxes)
        if new_player_pos in new_boxes:
            new_box_pos = (new_player_pos[0] + dx, new_player_pos[1] + dy)
            if new_box_pos in self.walls or new_box_pos in new_boxes:
                return False
        return True

    def copy_move(self, direction: Direction):
        new = Sokoban(self.level)
        if new.move(direction):
            return new
        return None

