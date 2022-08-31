from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QPixmap

from hf_vision.core import SignalSlotCachable, ThreadBase


class VisionSourceInfo(object):
    def __init__(self, source=None, width=0, height=0, fps=0):
        super().__init__()
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps


class VisionCVData(object):
    def __init__(self, success, cv_image):
        super().__init__()
        self.success = success
        self.cv_image = cv_image


class VisionSourceBaseProvider(ThreadBase, SignalSlotCachable):
    _frame_update_signal = pyqtSignal(QPixmap)
    _cv_frame_update_signal = pyqtSignal(VisionCVData)
    _video_info_update_signal = pyqtSignal(VisionSourceInfo)

    def connect_pixel_image_signal(self, slot):
        self.connect_signal_slot(self._frame_update_signal, slot)

    def connect_cv_image_signal(self, slot):
        self.connect_signal_slot(self._cv_frame_update_signal, slot)

    def connect_video_info_signal(self, slot):
        self.connect_signal_slot(self._video_info_update_signal, slot)

    def disconnect_signal(self):
        self.disconnect_all()
