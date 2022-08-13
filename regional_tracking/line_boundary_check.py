import uuid

import numpy as np
from numpy import linalg as LA

from regional_tracking.line_interset_util import Point


class BoundaryLine:
    def __init__(self, line=(0, 0, 0, 0)):
        self.uuid = uuid.uuid4()
        self.p0 = Point(line[0], line[1])
        self.p1 = Point(line[2], line[3])
        self.color = (0, 255, 255)
        self.line_thickness = 2
        self.textColor = (0, 255, 255)
        self.textSize = 4
        self.text_thickness = 2
        self.count1 = 0
        self.count2 = 0

    def setIntersect(self, flag: bool):
        self.color = (0, 0, 255) if flag else (0, 255, 255)

    def resetIntersect(self):
        self.setIntersect(flag=False)


# ---------------------------------------------
# Checking boundary line crossing detection

# def line(p1, p2):
#     A = (p1[1] - p2[1])
#     B = (p2[0] - p1[0])
#     C = (p1[0] * p2[1] - p2[0] * p1[1])
#     return A, B, -C


# Calcuate the coordination of intersect point of line segments - 線分同士が交差する座標を計算
# def calcIntersectPoint(line1p1, line1p2, line2p1, line2p2):
#     L1 = line(line1p1, line1p2)
#     L2 = line(line2p1, line2p2)
#     D = L1[0] * L2[1] - L1[1] * L2[0]
#     Dx = L1[2] * L2[1] - L1[1] * L2[2]
#     Dy = L1[0] * L2[2] - L1[2] * L2[0]
#     x = Dx / D
#     y = Dy / D
#     return x, y


# Check if line segments intersect - 判断线段是否相交，这个算法是错误的。当一个点在另外一个条线段上时，该算法有问题
# def checkIntersect(p1, p2, p3, p4) -> bool:
#     tc1 = (p1[0] - p2[0]) * (p3[1] - p1[1]) + (p1[1] - p2[1]) * (p1[0] - p3[0])
#     tc2 = (p1[0] - p2[0]) * (p4[1] - p1[1]) + (p1[1] - p2[1]) * (p1[0] - p4[0])
#     td1 = (p3[0] - p4[0]) * (p1[1] - p3[1]) + (p3[1] - p4[1]) * (p3[0] - p1[0])
#     td2 = (p3[0] - p4[0]) * (p2[1] - p3[1]) + (p3[1] - p4[1]) * (p3[0] - p2[0])
#     return tc1 * tc2 < 0 and td1 * td2 < 0


# convert a line to a vector
# line(point1)-(point2)
def line_vectorize(point1, point2):
    a = point2[0] - point1[0]
    b = point2[1] - point1[1]
    return [a, b]


# Calculate the angle made by two line segments - 線分同士が交差する角度を計算
# point = (x,y)
# line1(point1)-(point2), line2(point3)-(point4)
def calc_vector_angle(point1, point2, point3, point4):
    u = np.array(line_vectorize(point1, point2))
    v = np.array(line_vectorize(point3, point4))
    i = np.inner(u, v)
    n = LA.norm(u) * LA.norm(v)
    c = i / n
    a = np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0)))
    if u[0] * v[1] - u[1] * v[0] < 0:
        return a
    else:
        return 360 - a


# Test whether the test_point is in the polygon or not - 指定の点がポリゴン内に含まれるかどうかを判定
# test_point = (x,y)
# polygon = collection of points  [ (x0,y0), (x1,y1), (x2,y2) ... ]
def point_polygon_test(polygon, test_point) -> bool:
    if len(polygon) < 3:
        return False
    prev_point = polygon[-1]  # Use the last point as the starting point to close the polygon
    line_count = 0
    for point in polygon:
        # Check if Y coordinate of the test point is in range
        if min(prev_point[1], point[1]) <= test_point[1] <= max(prev_point[1], point[1]):
            # delta_x / delta_y
            gradient = (point[0] - prev_point[0]) / (point[1] - prev_point[1])
            # Calculate X coordinate of a line
            line_x = prev_point[0] + (test_point[1] - prev_point[1]) * gradient
            if line_x < test_point[0]:
                line_count += 1
        prev_point = point
    # Check how many lines exist on the left to the test_point
    included = True if line_count % 2 == 1 else False
    return included
