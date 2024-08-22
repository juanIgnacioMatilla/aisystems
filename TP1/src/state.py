from TP1.src.sokoban import Direction


class State:
    def __init__(self, player_pos, boxes):
        self.player_pos = player_pos
        self.boxes = tuple(
            sorted(boxes)
        )  # Tupla ordenada para garantizar unicidad en el set

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

    def is_blocked(self, walls, targets):
        set_walls = set(walls)
        set_targets = set(targets)

        for bx, by in self.boxes:
            # Case 1: Box in corner (and not on target)
            if (bx, by) not in set_targets and (
                    ((bx - 1, by) in set_walls and (bx, by - 1) in set_walls) or
                    ((bx + 1, by) in set_walls and (bx, by - 1) in set_walls) or
                    ((bx - 1, by) in set_walls and (bx, by + 1) in set_walls) or
                    ((bx + 1, by) in set_walls and (bx, by + 1) in set_walls)
            ):
                return True

            # Case 2: Wall-Box U-shape deadlock
            # Check for a wall-box pair on the LEFT, RIGHT, ABOVE, or BELOW and U-shape formed by wall-wall pairs

            # Case wall-box pair on the LEFT
            dx = -1
            dy = 0
            if (bx + dx, by + dy) in set_walls:

                # Check for wall-wall pair above and below the box
                i = 1
                above_wall_pair = False
                while (bx + dx, by - i) in set_walls:
                    if (bx, by - i) in set_walls:
                        above_wall_pair = True
                        break
                    i += 1

                i = 1
                below_wall_pair = False
                while (bx + dx, by + i) in set_walls:
                    if (bx, by + i) in set_walls:
                        below_wall_pair = True
                        break
                    i += 1

                if above_wall_pair and below_wall_pair:
                    # Count boxes and targets on the same axis
                    boxes_on_axis = sum(
                        1 for (bx2, _) in self.boxes if (bx2 == bx))
                    targets_on_axis = sum(
                        1 for (tx, _) in set_targets if (tx == bx))

                    if boxes_on_axis > targets_on_axis:
                        return True

            # Case wall-box pair on the RIGHT
            dx = 1
            dy = 0
            if (bx + dx, by + dy) in set_walls:

                # Check for wall-wall pair above and below the box
                i = 1
                above_wall_pair = False
                while (bx + dx, by - i) in set_walls:
                    if (bx, by - i) in set_walls:
                        above_wall_pair = True
                        break
                    i += 1

                i = 1
                below_wall_pair = False
                while (bx + dx, by + i) in set_walls:
                    if (bx, by + i) in set_walls:
                        below_wall_pair = True
                        break
                    i += 1

                if above_wall_pair and below_wall_pair:
                    # Count boxes and targets on the same axis
                    boxes_on_axis = sum(
                        1 for (bx2, _) in self.boxes if (bx2 == bx))
                    targets_on_axis = sum(
                        1 for (tx, _) in set_targets if (tx == bx))

                    if boxes_on_axis > targets_on_axis:
                        return True

            # Case wall-box pair ABOVE
            dx = 0
            dy = -1
            if (bx + dx, by + dy) in set_walls:

                # Check for wall-wall pair left and right of the box
                i = 1
                left_wall_pair = False
                while (bx - i, by + dy) in set_walls:
                    if (bx - i, by) in set_walls:
                        left_wall_pair = True
                        break
                    i += 1

                i = 1
                right_wall_pair = False
                while (bx + i, by + dy) in set_walls:
                    if (bx + i, by) in set_walls:
                        right_wall_pair = True
                        break
                    i += 1

                if left_wall_pair and right_wall_pair:
                    # Count boxes and targets on the same axis
                    boxes_on_axis = sum(
                        1 for (_, by2) in self.boxes if (by2 == by))
                    targets_on_axis = sum(
                        1 for (_, ty) in set_targets if (ty == by))

                    if boxes_on_axis > targets_on_axis:
                        return True

            # Case wall-box pair BELOW
            dx = 0
            dy = 1
            if (bx + dx, by + dy) in set_walls:

                # Check for wall-wall pair left and right of the box
                i = 1
                left_wall_pair = False
                while (bx - i, by + dy) in set_walls:
                    if (bx - i, by) in set_walls:
                        left_wall_pair = True
                        break
                    i += 1

                i = 1
                right_wall_pair = False
                while (bx + i, by + dy) in set_walls:
                    if (bx + i, by) in set_walls:
                        right_wall_pair = True
                        break
                    i += 1

                if left_wall_pair and right_wall_pair:
                    # Count boxes and targets on the same axis
                    boxes_on_axis = sum(
                        1 for (_, by2) in self.boxes if (by2 == by))
                    targets_on_axis = sum(
                        1 for (_, ty) in set_targets if (ty == by))

                    if boxes_on_axis > targets_on_axis:
                        return True


        # Case 3: Two adjacent boxes with adjacent walls
        from itertools import combinations
        for (bx1, by1), (bx2, by2) in combinations(self.boxes, 2):
            if abs(bx1 - bx2) == 1 and by1 == by2:
                if ((bx1 - 1, by1) in set_walls and (bx2 - 1, by2) in set_walls) or \
                        ((bx1 + 1, by1) in set_walls and (bx2 + 1, by2) in set_walls):
                    if (bx1, by1) not in set_targets or (bx2, by2) not in set_targets:
                        return True

                if ((bx1, by1 - 1) in set_walls and (bx2, by2 + 1) in set_walls) or \
                        ((bx1, by1 + 1) in set_walls and (bx2, by2 - 1) in set_walls):
                    if (bx1, by1) not in set_targets or (bx2, by2) not in set_targets:
                        return True

            elif abs(by1 - by2) == 1 and bx1 == bx2:
                if ((bx1, by1 - 1) in set_walls and (bx2, by2 - 1) in set_walls) or \
                        ((bx1, by1 + 1) in set_walls and (bx2, by2 + 1) in set_walls):
                    if (bx1, by1) not in set_targets or (bx2, by2) not in set_targets:
                        return True

                if ((bx1 - 1, by1) in set_walls and (bx2 + 1, by2) in set_walls) or \
                        ((bx1 + 1, by1) in set_walls and (bx2 - 1, by2) in set_walls):
                    if (bx1, by1) not in set_targets or (bx2, by2) not in set_targets:
                        return True

        return False

    def is_blocked_extended(self, walls, targets):
        set_walls = set(walls)
        set_targets = set(targets)

        # not already blocked
        if not self.is_blocked(walls, targets):
            # 4 or more boxes
            if len(self.boxes) >= 4:

                # Case 4: Four boxes in a square
                # Helper function to check if four points form a square
                def is_square(bx1, by1, bx2, by2, bx3, by3, bx4, by4):
                    dists = [
                        (abs(bx1 - bx2) + abs(by1 - by2)),
                        (abs(bx1 - bx3) + abs(by1 - by3)),
                        (abs(bx1 - bx4) + abs(by1 - by4)),
                        (abs(bx2 - bx3) + abs(by2 - by3)),
                        (abs(bx2 - bx4) + abs(by2 - by4)),
                        (abs(bx3 - bx4) + abs(by3 - by4)),
                    ]
                    dists.sort()
                    # Check if there are 4 equal smaller distances (sides) and 2 equal larger distances (diagonals)
                    return (dists[0] == dists[1] == dists[2] == dists[3] == 1 and
                            dists[4] == dists[5] == 2)

                for bx1, by1 in self.boxes:
                    for bx2, by2 in self.boxes:
                        if (bx1, by1) == (bx2, by2) or abs(bx1 - bx2) + abs(by1 - by2) > 2:
                            continue
                        for bx3, by3 in self.boxes:
                            if abs(bx1 - bx3) + abs(by1 - by3) > 2 or abs(bx2 - bx3) + abs(by2 - by3) > 2 or (bx2, by2) == (
                                    bx3, by3) or (bx1, by1) == (bx3, by3):
                                continue
                            for bx4, by4 in self.boxes:
                                if abs(bx1 - bx4) + abs(by1 - by4) > 2 or abs(bx2 - bx4) + abs(by2 - by4) > 2 or abs(
                                        bx3 - bx4) + abs(by3 - by4) > 2 or (bx3, by3) == (bx4, by4) or (bx1, by1) == (
                                        bx4, by4) or (bx2, by2) == (bx4, by4):
                                    continue
                                if (bx1, by1) in set_targets and (bx2, by2) in set_targets and (
                                        bx3, by3) in set_targets and (bx4, by4) in set_targets:
                                    return False
                                if is_square(bx1, by1, bx2, by2, bx3, by3, bx4, by4):
                                    return True
