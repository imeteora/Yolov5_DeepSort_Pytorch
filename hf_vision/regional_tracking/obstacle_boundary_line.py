import uuid

import cv2
import numpy as np

from hf_vision.regional_tracking import Point, ObstacleInterface, ObstacleType, ObstacleState, draw_text, TextAlign, \
    is_point_on_line


class BoundaryLine(ObstacleInterface):
    def __init__(self, line):
        super().__init__(ObstacleType.boundary_line)
        self.uuid = uuid.uuid4()
        self._p0 = Point(line[0][0], line[0][1])
        self._p1 = Point(line[1][0], line[1][1])
        self._pts = np.array(line, dtype=np.int32)
        self._count0 = 0
        self._count1 = 0
        self._normal_color = (255, 0, 0)
        self._highlight_color = (0, 0, 255)
        self._text_color = (0, 255, 255)
        self._thickness = 2
        self._font_size = 4
        self._text_thickness = 2

    @property
    def p0(self) -> Point:
        return self._p0

    @property
    def p1(self) -> Point:
        return self._p1

    @property
    def __dot_size(self) -> int:
        return 8 if self.is_selected else 6

    def hit_test(self, pt: Point) -> bool:
        return is_point_on_line(self.p0, self.p1, pt)

    def draw(self, image):
        if self.obstacle_state == ObstacleState.normal:
            current_color = self.normal_color
        elif self.obstacle_state == ObstacleState.highlight:
            current_color = self.highlight_color
        elif self.obstacle_state == ObstacleState.disabled:
            current_color = self.disabled_color
        else:
            current_color = self.normal_color

        cv2.polylines(image, [self.np_pts], False, current_color, self.thickness)
        draw_text(image, str(self.count0), self._pts[0], TextAlign.Bottom | TextAlign.Left, cv2.FONT_HERSHEY_PLAIN,
                  self.font_size, self.text_color, self.text_thickness)
        cv2.circle(image, self.np_pts[0], self.__dot_size,
                   (self.normal_color if not self.is_selected else self.highlight_color), -1)

        draw_text(image, str(self.count1), self._pts[1], TextAlign.Bottom | TextAlign.Right, cv2.FONT_HERSHEY_PLAIN,
                  self.font_size, self.text_color, self.text_thickness)
        cv2.circle(image, self.np_pts[1], self.__dot_size,
                   (self.normal_color if not self.is_selected else self.highlight_color), -1)
