from collections import deque

from TP1.src.sokoban import Direction


class State:
    def __init__(self, player_pos, boxes):
        self.player_pos = player_pos
        self.boxes = tuple(sorted(boxes))  # Tupla ordenada para garantizar unicidad en el set

    def __eq__(self, other):
        return self.player_pos == other.player_pos and self.boxes == other.boxes

    def __hash__(self):
        return hash((self.player_pos, self.boxes))

    def __str__(self):
        return f"Player Position: {self.player_pos}, Boxes: {self.boxes}"

    def __repr__(self):
        return f"State(player_pos={self.player_pos}, boxes={self.boxes})"

    def can_move(self, direction: Direction, walls):
        x, y = direction.value
        new_player_pos = (self.player_pos[0] + x, self.player_pos[1] + y)
        if new_player_pos in walls:
            return False
        if new_player_pos in self.boxes:
            new_box_pos = (new_player_pos[0] + x, new_player_pos[1] + y)
            if new_box_pos in walls or new_box_pos in self.boxes:
                return False
        return True

    def copy_move(self, direction: Direction, walls):
        if not self.can_move(direction, walls):
            return None
        x, y = direction.value
        new_player_pos = (self.player_pos[0] + x, self.player_pos[1] + y)
        new_boxes = set(self.boxes)
        if new_player_pos in new_boxes:
            new_box_pos = (new_player_pos[0] + x, new_player_pos[1] + y)
            new_boxes.remove(new_player_pos)
            new_boxes.add(new_box_pos)
        return State(new_player_pos, new_boxes)

    def is_completed(self, targets):
        return set(self.boxes) == targets