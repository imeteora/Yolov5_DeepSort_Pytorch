# ------------------------------------
# Object tracking
import time
import uuid

import numpy as np

from hf_vision.regional_tracking import Point


class RawObject(object):
    def __init__(self, pos, feature, id: int = -1):
        super().__init__()
        self.feature = feature
        self.id = id
        self.trajectory = []
        self.time = time.monotonic()
        self.pos = pos  # (left, top, right, bottom)

    @property
    def anchor_pt(self) -> [int]:
        p0 = (self.left + self.right) // 2
        p1 = int(self.top + round(self.height * 0.92))
        # p1 = (self.top + self.bottom) // 2
        return [p0, p1]

    @property
    def left(self) -> int:
        return int(round(self.pos[0]))

    @property
    def top(self) -> int:
        return int(round(self.pos[1]))

    @property
    def right(self) -> int:
        return int(round(self.pos[2]))

    @property
    def bottom(self) -> int:
        return int(round(self.pos[3]))

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


class BoundaryLine:
    def __init__(self, line=(0, 0, 0, 0)):
        self.uuid = uuid.uuid4()
        self.p0 = Point(line[0], line[1])
        self.p1 = Point(line[2], line[3])
        self.color = (0, 255, 255)
        self.line_thickness = 2
        self.text_color = (0, 255, 255)
        self.font_size = 4
        self.text_thickness = 2
        self.count1 = 0
        self.count2 = 0

    def setIntersect(self, flag: bool):
        self.color = (0, 0, 255) if flag else (0, 255, 255)

    def resetIntersect(self):
        self.setIntersect(flag=False)


# ------------------------------------
# Area intrusion detection
class Area:
    def __init__(self, contour):
        self.contour = np.array(contour, dtype=np.int32)
        self.count = 0
