import time

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QImage

from hf_vision.qt import utils
from vision_tracker_config import VisionTrackingConfig


def _time_interval() -> int:
    return int(round(time.time() * 1000))


class VisionSourceInfo(object):
    def __init__(self, source, width, height, fps):
        super().__init__()
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps


class VisionSourceProvider(QThread):
    _frame_update_signal = pyqtSignal(QPixmap)  # pyqtSignal(np.ndarray)
    _cv_frame_update_signal = pyqtSignal(np.ndarray)
    _video_info_update_signal = pyqtSignal(VisionSourceInfo)

    def set_video_file(self, path: str):
        self.video_src_path = path

    def connect_pixel_image_signal(self, slot):
        if slot is None:
            return
        self._frame_update_signal.connect(slot)

    def connect_cv_image_signal(self, slot):
        if slot is None:
            return
        self._cv_frame_update_signal.connect(slot)

    def connect_video_info_signal(self, slot):
        if slot is None:
            return
        self._video_info_update_signal.connect(slot)

    def disconnect_signal(self):
        self._frame_update_signal.disconnect()
        self._cv_frame_update_signal.disconnect()
        self._video_info_update_signal.disconnect()

    def __init__(self, config: VisionTrackingConfig):
        super().__init__()
        self._video_frame_time = None
        self._video_fps = None
        self._video_cap = None
        self.video_src_path = None
        self._is_running = False
        self.config = config
        self.display_width = 1920
        self.display_height = 1080

    def run(self):
        self._is_running = True
        self._video_cap = cv2.VideoCapture(self.video_src_path)
        self._video_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.display_width = self._video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.display_height = self._video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self._video_fps = self._video_cap.get(cv2.CAP_PROP_FPS)
        self._video_frame_time = 1000 / self._video_fps

        self._video_info_update_signal.emit(
            VisionSourceInfo(self.video_src_path, self.display_width, self.display_height, self._video_fps))

        _latest_frame_time = _time_interval()
        success, cap = True, None
        while self._is_running and success:
            _current_frame_time = _time_interval()
            if _current_frame_time < _latest_frame_time + self._video_frame_time:
                time.sleep(0)  # let cpu yield a while.
                continue
            _latest_frame_time = _current_frame_time

            success, cap = self._video_cap.read()
            if success:
                self._cv_frame_update_signal.emit(cap)  # vid frame img in numpy format
                if not self.config.using_tracker_vision:
                    cap = utils.convert_cv_qt(cv_img=cap, width=self.display_width, height=self.display_height)
                    self._frame_update_signal.emit(cap)

    def stop(self):
        self._is_running = False


