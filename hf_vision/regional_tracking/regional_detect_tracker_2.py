import time
from typing import List

import numpy as np
from numpy import linalg as LA

from hf_vision.regional_tracking import ObstacleInterface, ObstacleType, line_intersect, \
    ObstacleState, Point
from hf_vision.regional_tracking.raw_object import RawObject
from hf_vision.regional_tracking.regional_detect_tracker import RegionalDetectTracker
from hf_vision.regional_tracking.tracking_object import TrackingObject


class RegionalDetectTrackerM2(RegionalDetectTracker):

    @property
    def objects(self) -> [RawObject]:
        return [obj for (obj_id, obj) in self.object_db.items()]

    def __init__(self, conf_thres: float = 0.5):
        super().__init__()
        self.object_db = None
        self.timeout = 3  # sec
        self.conf_thres = conf_thres
        self.clear_db()

    def clear_db(self):
        self.object_db = {}

    def evictTimeoutObjectFromDB(self):
        now = time.monotonic()
        self.object_db = {key: val for key, val in self.object_db.items() if val.time + self.timeout >= now}

    def try_tracking(self, pos: List[int], id: int, feature=None, conf: float = 0.5):
        if conf < self.conf_thres:
            return

        if found := self.object_with(obj_id=id):
            found.update(pos=pos)
        else:
            new_obj = TrackingObject(pos, feature=feature, id=id)
            self.append_object(obj=new_obj)

    def object_with(self, obj_id: int) -> any:
        return self.object_db[obj_id] \
            if obj_id in self.object_db \
            else None

    def append_object(self, obj):
        if obj.id in self.object_db:
            return

        obj.time = time.monotonic()
        obj.trajectory = [obj.anchor_pt]
        self.object_db[obj.id] = obj

    def trackObjects(self, objects: List[RawObject]):
        # deprecated method in this M2 Tracker
        pass

    def tracking(self, obstacles: [ObstacleInterface]):
        # boundary lines
        self.__checking_boundary_lines_crossing(obstacles=obstacles)
        # regional area
        self.__checking_area_intersection(obstacles=obstacles)

    def drawTrajectory(self, img, objects):
        super().drawTrajectory(img=img, objects=objects)

    def obstacles_with_type(self, type: ObstacleType, obstacles: [ObstacleInterface]) -> [ObstacleInterface]:
        return [obstacle for obstacle in obstacles if obstacle.type == type]

    def __checking_boundary_lines_crossing(self, obstacles: [ObstacleInterface]):
        all_lines = self.obstacles_with_type(type=ObstacleType.boundary_line, obstacles=obstacles)
        if all_lines is None or len(all_lines) == 0:
            return
        self.__check_boundary_lines_crosses(all_lines, self.objects)

    def __checking_area_intersection(self, obstacles: [ObstacleInterface]):
        all_areas = self.obstacles_with_type(type=ObstacleType.area, obstacles=obstacles)
        if all_areas is None or len(all_areas) == 0:
            return
        self.__check_area_intrusion(all_areas, self.objects)

    # BOUNDARY LINES CHECKING -----------------------------------------------------------------------

    # convert a line to a vector
    # line(point1)-(point2)
    def __line_vectorize(self, point1, point2):
        a = point2[0] - point1[0]
        b = point2[1] - point1[1]
        return [a, b]

    # Calculate the angle made by two line segments - 線分同士が交差する角度を計算
    # point = (x,y)
    # line1(point1)-(point2), line2(point3)-(point4)
    def __calc_vector_angle(self, point1, point2, point3, point4):
        u = np.array(self.__line_vectorize(point1, point2))
        v = np.array(self.__line_vectorize(point3, point4))
        i = np.inner(u, v)
        n = LA.norm(u) * LA.norm(v)
        c = i / n
        a = np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0)))
        return a \
            if u[0] * v[1] - u[1] * v[0] < 0 \
            else 360 - a

    # in: boundary_line = boundaryLine class object
    #     trajectory   = (x1, y1, x2, y2)
    def __check_line_cross(self, boundary_line, trajectory) -> bool:
        traj_p0 = trajectory[0]  # Trajectory of an object
        traj_p1 = trajectory[1]
        bLine_p0 = boundary_line.p0  # Boundary line
        bLine_p1 = boundary_line.p1
        intersect = line_intersect(traj_p0, traj_p1, bLine_p0, bLine_p1)  # Check if intersect or not

        if intersect is True:
            # boundary_line.set_intersect(flag=intersect)
            boundary_line.obstacle_state = ObstacleState.highlight
            # Calculate angle between trajectory and boundary line
            angle = self.__calc_vector_angle((traj_p0.x, traj_p0.y),
                                             (traj_p1.x, traj_p1.y),
                                             (bLine_p0.x, bLine_p0.y),
                                             (bLine_p1.x, bLine_p1.y))
            if angle < 180:
                boundary_line.count0 += 1
            else:
                boundary_line.count1 += 1

        return intersect

    # Multiple lines cross checking
    def __check_boundary_lines_crosses(self, boundaryLines, objects):
        objs_with_trajs = [obj for obj in objects if len(obj.trajectory) > 1]
        for obj in objs_with_trajs:
            lines_not_crossed = [bline for bline in boundaryLines if bline.uuid not in obj.crossed_lines]
            for line in lines_not_crossed:
                for idx in range(len(obj.trajectory) - 1):
                    p0 = Point.point(obj.trajectory[idx])
                    p1 = Point.point(obj.trajectory[idx + 1])
                    if self.__check_line_cross(line, [p0, p1]):
                        obj.crossed_lines.append(line.uuid)
                        break

    # AREA CHECKING -----------------------------------------------------------------------------

    # Area intrusion check
    def __check_area_intrusion(self, areas, objects):
        # global audio_enable_flag
        # global sound_thread_warning
        for area in areas:
            area.count0 = 0
            for obj in objects:
                if self.__point_polygon_test(area.np_pts, obj.anchor_pt):
                    area.count0 += 1

    # Test whether the test_point is in the polygon or not - 指定の点がポリゴン内に含まれるかどうかを判定
    # test_point = (x,y)
    # polygon = collection of points  [ (x0,y0), (x1,y1), (x2,y2) ... ]
    def __point_polygon_test(self, polygon, test_point) -> bool:
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


# Draw multiple boundary lines
def draw_boundary_lines(img, lines):
    for line in lines:
        line.draw(img)


def resetLineCrosses(boundaryLines):
    for line in boundaryLines:
        line.obstacle_state = ObstacleState.normal


# Draw areas (polygons)
def draw_polylines(img, areas):
    [area.draw(img) for area in areas]
