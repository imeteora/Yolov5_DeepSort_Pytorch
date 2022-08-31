import uuid
from enum import Enum

import numpy as np

from hf_vision.regional_tracking import Selectable, Point
from hf_vision.regional_tracking.renderable import Renderable


class ObstacleType(Enum):
    boundary_line = 0
    area = boundary_line + 1


class ObstacleState(Enum):
    normal = 0
    highlight = normal + 1
    disabled = highlight + 1


class ObstacleInterface(Selectable, Renderable):

    def __init__(self, type: ObstacleType):
        super().__init__()
        self.uuid = uuid.uuid4()
        self.type = type

        self._state = ObstacleState.normal
        self._pts = None
        self._count0 = 0
        self._count1 = 0
        self._font_size = 4
        self._text_color = (0, 255, 255)
        self._text_thickness = 2
        self._thickness = 1
        self._normal_color = (128, 128, 128)
        self._highlight_color = (255, 255, 255)
        self._disabled_color = (196, 196, 196)

    @property
    def obstacle_state(self) -> ObstacleState:
        return self._state

    @obstacle_state.setter
    def obstacle_state(self, new_value):
        self._state = new_value

    @property
    def np_pts(self) -> np.array:
        return self._pts

    @property
    def normal_color(self) -> tuple[int, int, int]:
        return self._normal_color

    @normal_color.setter
    def normal_color(self, new_value: tuple[int, int, int]):
        self._normal_color = new_value

    @property
    def highlight_color(self) -> tuple[int, int, int]:
        return self._highlight_color

    @highlight_color.setter
    def highlight_color(self, new_value):
        self._highlight_color = new_value

    @property
    def disabled_color(self) -> tuple[int, int, int]:
        return self._disabled_color

    @disabled_color.setter
    def disabled_color(self, new_value):
        self._disabled_color = new_value

    @property
    def text_color(self) -> tuple[int, int, int]:
        return self._text_color

    @property
    def thickness(self) -> int:
        return self._thickness

    @thickness.setter
    def thickness(self, new_value):
        self._thickness = new_value if new_value > 0 else 1

    @property
    def font_size(self) -> int:
        return self._font_size

    @font_size.setter
    def font_size(self, new_value):
        self._font_size = new_value

    @property
    def text_thickness(self) -> int:
        return self._text_thickness

    @text_thickness.setter
    def text_thickness(self, new_value):
        self._text_thickness = new_value

    @property
    def count0(self) -> int:
        return self._count0

    @count0.setter
    def count0(self, new_value):
        self._count0 = new_value

    @property
    def count1(self) -> int:
        return self._count1

    @count1.setter
    def count1(self, new_value):
        self._count1 = new_value

    def hit_test(self, pt: Point) -> bool:
        assert (False, 'must be implemented in sub-classes')
        return False

    def draw(self, image):
        assert (False, "must be implemented in sub-class.")
        pass
