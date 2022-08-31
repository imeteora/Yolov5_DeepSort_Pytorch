import cv2
import numpy as np

from hf_vision.regional_tracking import draw_text, TextAlign, ObstacleType, Point
from hf_vision.regional_tracking.obstacle_interface import ObstacleInterface


class Area(ObstacleInterface):
    def __init__(self, contour):
        super().__init__(ObstacleType.area)
        self._pts = np.array(contour, dtype=np.int32)
        self._count0 = 0
        self._count1 = 0
        self._normal_color = (255, 0, 0)
        self._highlight_color = (0, 0, 255)
        self._text_color = (0, 255, 255)
        self._font_size = 4
        self._text_thickness = 2
        self._thickness = 2

    def hit_test(self, pt: Point) -> bool:
        return False

    def draw(self, image):
        if self.count0 <= 0:
            color = self.normal_color
        else:
            color = self.highlight_color

        cv2.polylines(image, [self.np_pts], True, color, self.thickness)
        draw_text(image, str(self.count0), (self.np_pts[0][0], self.np_pts[0][1]),
                  TextAlign.Left | TextAlign.Bottom,
                  cv2.FONT_HERSHEY_PLAIN, self.font_size,
                  color, self.text_thickness)
