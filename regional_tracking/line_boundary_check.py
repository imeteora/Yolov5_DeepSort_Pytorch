import numpy as np
from numpy import linalg as LA

from regional_tracking.line_interset_util import line_intersect, Point


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
    return a \
        if u[0] * v[1] - u[1] * v[0] < 0 \
        else 360 - a


# in: boundary_line = boundaryLine class object
#     trajectory   = (x1, y1, x2, y2)
def checkLineCross(boundary_line, trajectory) -> bool:
    traj_p0 = trajectory[0]  # Trajectory of an object
    traj_p1 = trajectory[1]
    bLine_p0 = boundary_line.p0  # Point(boundary_line.p0[0], boundary_line.p0[1])  # Boundary line
    bLine_p1 = boundary_line.p1  # Point(boundary_line.p1[0], boundary_line.p1[1])
    intersect = line_intersect(traj_p0, traj_p1, bLine_p0, bLine_p1)  # Check if intersect or not

    if intersect is True:
        boundary_line.setIntersect(flag=intersect)
        angle = calc_vector_angle((traj_p0.x, traj_p0.y),
                                  (traj_p1.x, traj_p1.y),
                                  (bLine_p0.x, bLine_p0.y),
                                  (bLine_p1.x, bLine_p1.y))  # Calculate angle between trajectory and boundary line
        if angle < 180:
            boundary_line.count1 += 1
        else:
            boundary_line.count2 += 1
        # cx, cy = calcIntersectPoint(traj_p0, traj_p1, bLine_p0, bLine_p1) # Calculate the intersect coordination

    return intersect


# Multiple lines cross check
def checkLineCrosses(boundaryLines, objects):
    objs_with_trajs = [obj for obj in objects if len(obj.trajectory) > 1]
    for obj in objs_with_trajs:
        lines_not_crossed = [bline for bline in boundaryLines if bline.uuid not in obj.crossed_lines]
        for line in lines_not_crossed:
            for idx in range(len(obj.trajectory) - 1):
                p0 = Point.point(obj.trajectory[idx])
                p1 = Point.point(obj.trajectory[idx + 1])
                if checkLineCross(line, [p0, p1]):
                    obj.crossed_lines.append(line.uuid)
                    break


def resetLineCrosses(boundaryLines):
    for line in boundaryLines:
        line.resetIntersect()
