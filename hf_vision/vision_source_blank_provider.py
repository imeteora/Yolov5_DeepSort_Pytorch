import numpy as np
from PyQt5.QtCore import pyqtSlot

from hf_vision import VisionSourceBaseProvider


class VisionSourceBlankProvider(VisionSourceBaseProvider):
    def __init__(self):
        super().__init__()
        self.display_width = None
        self.display_height = None
        self._color_channel_val = 255
        self._is_running = False
        self._img_buffer = None
        self._renew_canvas_blank_img()

    def main(self) -> bool:
        self._cv_frame_update_signal.emit(self._img_buffer)
        return True

    def _refresh(self):
        self._renew_canvas_blank_img()

    @pyqtSlot((int, int))
    def resize_slot(self, new_value):
        self.display_width = new_value[0]
        self.display_height = new_value[1]
        self._refresh()

    @pyqtSlot(int)
    def color_slot(self, new_value):
        self._color_channel_val = new_value
        self._refresh()

    def _renew_canvas_blank_img(self):
        self._img_buffer = self._color_channel_val * np.ones(shape=(self.display_height, self.display_width, 3),
                                                             dtype=np.uint8)
