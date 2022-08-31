import sys
from enum import Enum


class ObstacleType(Enum):
    none = 0
    boundary_line = none + 1
    area = boundary_line + 1

    def max_pts_count(self) -> int:
        if self is ObstacleType.boundary_line:
            return 2
        else:
            return sys.maxsize
