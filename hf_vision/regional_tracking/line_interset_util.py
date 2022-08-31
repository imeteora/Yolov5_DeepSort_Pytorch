import math
from typing import List, TypeVar

S = TypeVar('S', bound='Size')


class Size:
    @classmethod
    def zero(cls) -> S:
        return cls(0, 0)

    def __init__(self, width, height):
        self._width = width
        self._height = height

    def __str__(self):
        return f'Size: [{self.width}, {self.height}]'

    @property
    def width(self) -> int:
        return self._width

    @width.setter
    def width(self, new_value):
        self._width = new_value if new_value is not None else 0

    @property
    def height(self) -> int:
        return self._height

    @height.setter
    def height(self, new_value):
        self._height = new_value if new_value is not None else 0


P = TypeVar('P', bound='Point')


class Point:
    @classmethod
    def zero(cls) -> P:
        return cls(0, 0)

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f'Point: [{self.x}, {self.y}]'

    def scale(self, factor: float) -> P:
        return Point(self.x * factor, self.y * factor)

    @property
    def raw_pt(self) -> (int, int):
        return self.x, self.y

    @classmethod
    def point(cls, pts: List[int]):
        return cls(x=pts[0], y=pts[1])


R = TypeVar('R', bound='Rect')


class Rect:
    @classmethod
    def zero(cls) -> R:
        return cls(origin=Point.zero(), size=Size.zero())

    def __init__(self, origin: Point, size: Size):
        self.origin = origin
        self.size = size

    def __str__(self):
        return f'Rect: [{self.origin}, {self.size}]'

    @property
    def x(self) -> int:
        return self.origin.x

    @property
    def y(self) -> int:
        return self.origin.y

    @property
    def width(self):
        return self.size.width

    @property
    def height(self):
        return self.size.height

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @x.setter
    def x(self, new_value):
        self.origin.x = new_value if new_value is not None else 0

    @y.setter
    def y(self, new_value):
        self.origin.y = new_value if new_value is not None else 0

    @width.setter
    def width(self, new_value):
        self.size.width = new_value if new_value is not None else 0

    @height.setter
    def height(self, new_value):
        self.size.height = new_value if new_value is not None else 0


def ccw(A, B, C):
    return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)


def line_intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def pt_dist(A: Point, B: Point) -> float:
    return math.sqrt((B.x - A.x)**2 + (B.y - A.y)**2)


def is_point_on_line(A: Point, B: Point, C: Point) -> bool:
    thres = 1
    return -thres <= pt_dist(A, C) + pt_dist(C, B) - pt_dist(A, B) <= thres

