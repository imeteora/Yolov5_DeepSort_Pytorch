# ------------------------------------
# Object tracking
import time


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
