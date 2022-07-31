import cv2
import numpy as np

from regional_tracking import Point, line_intersect
from regional_tracking.line_boundary_check import pointPolygonTest, calcVectorAngle


# ------------------------------------
# Area intrusion detection

class area:
    def __init__(self, contour):
        self.contour = np.array(contour, dtype=np.int32)
        self.count = 0


# Area intrusion check
def checkAreaIntrusion(areas, objects):
    # global audio_enable_flag
    # global sound_thread_warning
    for area in areas:
        area.count = 0
        for obj in objects:
            if pointPolygonTest(area.contour, obj.anchor_pt):
                area.count += 1


# Draw areas (polygons)
def drawAreas(img, areas):
    for area in areas:
        if area.count > 0:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)
        cv2.polylines(img, [area.contour], True, color, 4)
        cv2.putText(img, str(area.count), (area.contour[0][0], area.contour[0][1]), cv2.FONT_HERSHEY_PLAIN, 4, color, 2)


# in: boundary_line = boundaryLine class object
#     trajectory   = (x1, y1, x2, y2)
def checkLineCross(boundary_line, trajectory):
    traj_p0 = Point(trajectory[0], trajectory[1])  # Trajectory of an object
    traj_p1 = Point(trajectory[2], trajectory[3])
    bLine_p0 = Point(boundary_line.p0[0], boundary_line.p0[1])  # Boundary line
    bLine_p1 = Point(boundary_line.p1[0], boundary_line.p1[1])
    intersect = line_intersect(traj_p0, traj_p1, bLine_p0, bLine_p1)  # Check if intersect or not

    if intersect is True:
        boundary_line.setIntersect(flag=intersect)
        angle = calcVectorAngle((traj_p0.x, traj_p0.y),
                                (traj_p1.x, traj_p1.y),
                                (bLine_p0.x, bLine_p0.y),
                                (bLine_p1.x, bLine_p1.y))  # Calculate angle between trajectory and boundary line
        if angle < 180:
            boundary_line.count1 += 1
        else:
            boundary_line.count2 += 1
        # cx, cy = calcIntersectPoint(traj_p0, traj_p1, bLine_p0, bLine_p1) # Calculate the intersect coordination


# Multiple lines cross check
def checkLineCrosses(boundaryLines, objects):
    for obj in objects:
        traj = obj.trajectory
        if len(traj) > 1:
            p0 = traj[-2]
            p1 = traj[-1]
            for line in boundaryLines:
                checkLineCross(line, [p0[0], p0[1], p1[0], p1[1]])


def resetLineCrosses(boundaryLines):
    for line in boundaryLines:
        line.resetIntersect()
