class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @classmethod
    def point(cls, pts: [int]):
        return cls(x=pts[0], y=pts[1])


def ccw(A, B, C):
    return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)


def line_intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
