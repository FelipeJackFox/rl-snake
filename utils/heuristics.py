from __future__ import annotations
from typing import Tuple

# Absolute directions: 0=up,1=right,2=down,3=left
DIRS = [(0,-1),(1,0),(0,1),(-1,0)]

# Relative actions: 0=left,1=straight,2=right
REL_FROM_TO = {
    (0, 0): 3, (0, 1): 0, (0, 2): 1, (0, 3): 2,
    (1, 0): 2, (1, 1): 1, (1, 2): 0, (1, 3): 3,
    (2, 0): 1, (2, 1): 2, (2, 2): 3, (2, 3): 0,
    (3, 0): 0, (3, 1): 3, (3, 2): 2, (3, 3): 1,
}

def greedy_relative_action(head_dir: int, head_xy: Tuple[int,int], food_xy: Tuple[int,int]) -> int:
    hx, hy = head_xy
    fx, fy = food_xy
    if abs(fx - hx) > abs(fy - hy):
        best_dir = 1 if fx > hx else 3
    else:
        best_dir = 2 if fy > hy else 0
    return REL_FROM_TO[(head_dir, best_dir)]
