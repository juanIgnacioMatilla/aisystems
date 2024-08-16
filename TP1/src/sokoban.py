from enum import Enum

class Symbol(Enum):
    WALL = '#'
    FREE = ' '
    BOX = '$'
    PLAYER = '@'
    TARGET = '.'
    BOX_ON_TARGET = '*'
    PLAYER_ON_TARGET = '+'


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
        for y, row in enumerate(self.level):
            for x, cell in enumerate(row):
                if (x, y) == self.player_pos:
                    print(Symbol.PLAYER, end='')
                elif (x, y) in self.boxes and (x, y) in self.targets:
                    print(Symbol.BOX_ON_TARGET, end='')
                elif (x, y) in self.boxes:
                    print(Symbol.BOX, end='')
                elif (x, y) in self.targets:
                    print(Symbol.TARGET, end='')
                elif (x, y) in self.walls:
                    print(Symbol.WALL, end='')
                else:
                    print(' ', end='')
            print()