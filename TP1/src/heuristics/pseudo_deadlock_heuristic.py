import sys
from itertools import combinations
from typing import Callable
from TP1.src.state import State


def pseudo_deadlock_heuristic(targets: set, walls: set) -> Callable[[State], float]:
    def f(state: State) -> float:
        penalty = 1
        set_walls = set(walls)
        set_targets = set(targets)

        # Case 1: Box in corner (and not on target)
        for bx, by in state.boxes:
            if (bx, by) not in set_targets and (
                    ((bx - 1, by) in set_walls and (bx, by - 1) in set_walls) or
                    ((bx + 1, by) in set_walls and (bx, by - 1) in set_walls) or
                    ((bx - 1, by) in set_walls and (bx, by + 1) in set_walls) or
                    ((bx + 1, by) in set_walls and (bx, by + 1) in set_walls)
            ):
                # DEADLOCK
                penalty += sys.float_info.max  # High penalty for box in corner

        # Case 2: Wall-Box U-shape deadlock
        for bx, by in state.boxes:

            # Case wall-box pair on the LEFT
            if (bx - 1, by) in set_walls:
                # Check for U-shape formed by wall-wall pairs above and below
                i = 1
                above_wall_wall_pair = False
                while (bx - 1, by - i) in set_walls:
                    if (bx, by - i) in set_walls:
                        # Found a wall-wall pair above
                        above_wall_wall_pair = True
                        break
                    i += 1

                i = 1
                below_wall_wall_pair = False
                while (bx - 1, by + i) in set_walls:
                    if (bx, by + i) in set_walls:
                        # Found a wall-wall pair below
                        below_wall_wall_pair = True
                        break
                    i += 1

                if above_wall_wall_pair and below_wall_wall_pair:
                    # Count boxes and targets on the same horizontal axis
                    boxes_on_axis = sum(1 for (bx2, _) in state.boxes if bx2 == bx)
                    targets_on_axis = sum(1 for (tx, _) in set_targets if tx == bx)

                    if boxes_on_axis > targets_on_axis:
                        # DEADLOCK
                        penalty += sys.float_info.max  # Assign a large penalty for this U-shape deadlock

            # Case wall-box pair on the RIGHT
            elif (bx + 1, by) in set_walls:
                # Check for U-shape formed by wall-wall pairs above and below
                i = 1
                above_wall_wall_pair = False
                while (bx + 1, by - i) in set_walls:
                    if (bx, by - i) in set_walls:
                        above_wall_wall_pair = True
                        break
                    i += 1

                i = 1
                below_wall_wall_pair = False
                while (bx + 1, by + i) in set_walls:
                    if (bx, by + i) in set_walls:
                        below_wall_wall_pair = True
                        break
                    i += 1

                if above_wall_wall_pair and below_wall_wall_pair:
                    # Count boxes and targets on the same horizontal axis
                    boxes_on_axis = sum(1 for (bx2, _) in state.boxes if bx2 == bx)
                    targets_on_axis = sum(1 for (tx, _) in set_targets if tx == bx)

                    if boxes_on_axis > targets_on_axis:
                        # DEADLOCK
                        penalty += sys.float_info.max

            # Case wall-box pair ABOVE
            if (bx, by - 1) in set_walls:
                # Check for U-shape formed by wall-wall pairs left and right
                i = 1
                left_wall_wall_pair = False
                while (bx - i, by - 1) in set_walls:
                    if (bx - i, by) in set_walls:
                        left_wall_wall_pair = True
                        break
                    i += 1

                i = 1
                right_wall_wall_pair = False
                while (bx + i, by - 1) in set_walls:
                    if (bx + i, by) in set_walls:
                        right_wall_wall_pair = True
                        break
                    i += 1

                if left_wall_wall_pair and right_wall_wall_pair:
                    # Count boxes and targets on the same vertical axis
                    boxes_on_axis = sum(1 for (_, by2) in state.boxes if by2 == by)
                    targets_on_axis = sum(1 for (_, ty) in set_targets if ty == by)

                    if boxes_on_axis > targets_on_axis:
                        # DEADLOCK
                        penalty += sys.float_info.max

            # Case wall-box pair BELOW
            elif (bx, by + 1) in set_walls:
                # Check for U-shape formed by wall-wall pairs left and right
                i = 1
                left_wall_wall_pair = False
                while (bx - i, by + 1) in set_walls:
                    if (bx - i, by) in set_walls:
                        left_wall_wall_pair = True
                        break
                    i += 1

                i = 1
                right_wall_wall_pair = False
                while (bx + i, by + 1) in set_walls:
                    if (bx + i, by) in set_walls:
                        right_wall_wall_pair = True
                        break
                    i += 1

                if left_wall_wall_pair and right_wall_wall_pair:
                    # Count boxes and targets on the same vertical axis
                    boxes_on_axis = sum(1 for (_, by2) in state.boxes if by2 == by)
                    targets_on_axis = sum(1 for (_, ty) in set_targets if ty == by)

                    if boxes_on_axis > targets_on_axis:
                        # DEADLOCK
                        penalty += sys.float_info.max

        # Case 3: Two adjacent boxes with adjacent walls
        for (bx1, by1), (bx2, by2) in combinations(state.boxes, 2):
            # Adjacent boxes
            if abs(bx1 - bx2) == 1 and by1 == by2:
                # Check for adjacent walls on the same side
                if ((bx1 - 1, by1) in set_walls and (bx2 - 1, by2) in set_walls) or \
                        ((bx1 + 1, by1) in set_walls and (bx2 + 1, by2) in set_walls):
                    if (bx1, by1) not in set_targets or (bx2, by2) not in set_targets:
                        # DEADLOCK
                        penalty += sys.float_info.max  # Penalty for boxes with shared wall

                # Adjacent boxes with walls on opposite sides
                if ((bx1, by1 - 1) in set_walls and (bx2, by2 + 1) in set_walls) or \
                        ((bx1, by1 + 1) in set_walls and (bx2, by2 - 1) in set_walls):
                    if (bx1, by1) not in set_targets or (bx2, by2) not in set_targets:
                        # DEADLOCK
                        penalty += sys.float_info.max

            elif abs(by1 - by2) == 1 and bx1 == bx2:
                # Check for adjacent walls on the same side
                if ((bx1, by1 - 1) in set_walls and (bx2, by2 - 1) in set_walls) or \
                        ((bx1, by1 + 1) in set_walls and (bx2, by2 + 1) in set_walls):
                    if (bx1, by1) not in set_targets or (bx2, by2) not in set_targets:
                        # DEADLOCK
                        penalty += sys.float_info.max  # Penalty for boxes with shared wall

                # Adjacent boxes with walls on opposite sides
                if ((bx1 - 1, by1) in set_walls and (bx2 + 1, by2) in set_walls) or \
                        ((bx1 + 1, by1) in set_walls and (bx2 - 1, by2) in set_walls):
                    if (bx1, by1) not in set_targets or (bx2, by2) not in set_targets:
                        # DEADLOCK
                        penalty += sys.float_info.max  # Penalty for adjacent boxes with walls on opposite sides
        return penalty

    return f

# def pseudo_deadlock_heuristic1(state: State, targets: set, walls: set) -> Callable[[State], float]:
#     def f(state: State = state) -> float:
#         penalty = 0
#
#         # Condition 1: Check if a wall immobilizes a box (axis of movement) with no available targets on that axis
#         for bx, by in state.boxes:
#
#             # Check horizontal axis (left and right)
#             if (
#                     ((bx - 1, by) in walls and (bx + 1, by) in walls) and
#                     all(tx != bx for tx, ty in targets)
#             ):
#                 penalty += 3  # Large penalty for pseudo-deadlock due to wall on horizontal axis
#
#             # Check vertical axis (up and down)
#             if (
#                     ((bx, by - 1) in walls and (bx, by + 1) in walls) and
#                     all(ty != by for tx, ty in targets)
#             ):
#                 penalty += 3  # Large penalty for pseudo-deadlock due to wall on vertical axis
#
#         # Condition 2: Check if a box blocks the movements of another box
#         for bx1, by1 in state.boxes:
#             for bx2, by2 in state.boxes:
#                 if (bx1, by1) != (bx2, by2):
#                     # Check if box1 blocks box2 on horizontal or vertical axis
#                     if bx1 == bx2 and abs(by1 - by2) == 1:  # Box2 is directly above/below Box1
#                         penalty += 1  # Penalty for potential blocking on the vertical axis
#                     elif by1 == by2 and abs(bx1 - bx2) == 1:  # Box2 is directly left/right of Box1
#                         penalty += 1  # Penalty for potential blocking on the horizontal axis
#
#         return penalty
#     return f
#
# def pseudo_deadlock_heuristic2(state: State, targets: set, walls: set) -> Callable[[State], float]:
#         def f(state: State = state) -> float:
#             penalty = 0
#
#             # Condition 1: Check if a box is blocked against a corner and is not on a target
#             for bx, by in state.boxes:
#                 if (bx, by) not in targets:
#                     if ((bx - 1, by) in walls and (bx, by - 1) in walls) or \
#                         ((bx + 1, by) in walls and (bx, by - 1) in walls) or \
#                         ((bx - 1, by) in walls and (bx, by + 1) in walls) or \
#                         ((bx + 1, by) in walls and (bx, by + 1) in walls):
#                         penalty += 50  # Penalty for being stuck in a corner
#
#             # Condition 2: Check if two boxes are blocking each other with walls on the sides
#             for bx1, by1 in state.boxes:
#                 for bx2, by2 in state.boxes:
#                     if (bx1, by1) != (bx2, by2):
#                         # Check if box1 and box2 are on the same horizontal line with walls on left and right
#                         if bx1 == bx2 and abs(by1 - by2) == 1:
#                             if (bx1 - 1, by1) in walls and (bx1 + 1, by1) in walls:
#                                 penalty += 2  # Penalty for potential blocking on horizontal axis
#                             if (bx1 - 1, by2) in walls and (bx1 + 1, by2) in walls:
#                                 penalty += 2  # Penalty for potential blocking on horizontal axis
#
#                         # Check if box1 and box2 are on the same vertical line with walls on top and bottom
#                         if by1 == by2 and abs(bx1 - bx2) == 1:
#                             if (bx1, by1 - 1) in walls and (bx1, by1 + 1) in walls:
#                                 penalty += 2  # Penalty for potential blocking on vertical axis
#                             if (bx2, by2 - 1) in walls and (bx2, by2 + 1) in walls:
#                                 penalty += 2  # Penalty for potential blocking on vertical axis
#
#             # Condition 3: Check if a wall immobilizes a box with no available targets on that axis
#             for bx, by in state.boxes:
#                 # Check horizontal axis (left and right)
#                 if (
#                         ((bx - 1, by) in walls and (bx + 1, by) in walls) and
#                         all(tx != bx for tx, ty in targets)
#                 ):
#                     penalty += 3  # Large penalty for pseudo-deadlock due to wall on horizontal axis
#
#                 # Check vertical axis (up and down)
#                 if (
#                         ((bx, by - 1) in walls and (bx, by + 1) in walls) and
#                         all(ty != by for tx, ty in targets)
#                 ):
#                     penalty += 3  # Large penalty for pseudo-deadlock due to wall on vertical axis
#
#             return penalty
#
#         return f
