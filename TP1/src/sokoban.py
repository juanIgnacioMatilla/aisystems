from enum import Enum

class Symbol(Enum):
    WALL = '#'
    FREE = ' '
    BOX = '$'
    PLAYER = '@'
    TARGET = '.'
    BOX_ON_TARGET = '='
    PLAYER_ON_TARGET = '!'


