import uuid

import numpy as np

from hf_vision.vision_obstacle_type import ObstacleType
from hf_vision.regional_tracking import Point, ObstacleInterface, BoundaryLine, Area


class VisionObstacleEditObject(object):
    @classmethod
    def none(cls) -> any:
        return cls(type=ObstacleType.none)

    def __init__(self, type: ObstacleType = ObstacleType.boundary_line, pts: [[int]] = None):
        super().__init__()
        self.uuid_id = uuid.uuid4()
        self.type = type
        self._force_done = False  # mark current object is finished editing, no more point allowed push in.
        self._pts = list(map(lambda e: Point(e[0], e[1]), pts)) \
            if pts is not None \
            else []

    def __len__(self):
        return len(self.pts)

    def __getitem__(self, item):
        return self.pts[item] if 0 <= item < len(self) else None

    def __setitem__(self, key, value):
        if self._force_done:
            return
        if 0 <= key < len(self):
            self.pts[key] = value

    def push(self, pt: Point):
        if self._force_done:
            return
        if self.type.max_pts_count() <= len(self):
            return
        self.pts.append(pt)

    def pop(self) -> Point:
        if self._force_done:
            return None

        result = self.pts[-1]
        self.pts = self.pts[:-1]
        return result

    @property
    def is_done(self) -> bool:
        return self._force_done or len(self) == self.type.max_pts_count()

    @is_done.setter
    def is_done(self, new_value: bool):
        self._force_done = new_value

    @property
    def pts(self) -> [Point]:
        return self._pts if not self.type == ObstacleType.none else []

    @pts.setter
    def pts(self, new_value):
        if not self.type == ObstacleType.none:
            self._pts = new_value

    @property
    def np_pts(self) -> np.array:
        pts_array = list(map(lambda pt: (pt.x, pt.y), self.pts))
        return np.array(pts_array, dtype=np.int32)

    @property
    def to_obstacle(self) -> ObstacleInterface:
        pts_array = [[p.x, p.y] for p in self.pts]
        if self.type == ObstacleType.boundary_line:
            return BoundaryLine(line=pts_array)
        elif self.type == ObstacleType.area:
            return Area(contour=pts_array)
        else:
            return None
