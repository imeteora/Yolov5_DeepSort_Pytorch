import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot

from hf_vision.editor_playback_state import EditorPlaybackState
from hf_vision import VisionSourceInfo, VisionSourceBaseProvider, VisionCVData
from hf_vision.core import ThreadState
from hf_vision.qt import utils
from vision_tracker_config import VisionTrackingConfig


class VisionSourceProvider(VisionSourceBaseProvider):
    _state_update_signal = pyqtSignal(ThreadState)

    def set_video_file(self, path: str):
        self.video_src_path = path

    def connect_thread_state_update_signal(self, slot):
        self.connect_signal_slot(self._state_update_signal, slot)

    def __init__(self, config: VisionTrackingConfig):
        super().__init__()
        self._video_frame_time = None
        self._video_fps = None
        self._video_cap = None
        self.video_src_path = None
        self.config = config
        self.display_width = 1920
        self.display_height = 1080
        self._cur_playback_state = EditorPlaybackState.stop
        self._last_cap = None

    def pre_main(self):
        self._cur_playback_state = EditorPlaybackState.start
        self._video_cap = cv2.VideoCapture(self.video_src_path)
        self._video_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.display_width = self._video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.display_height = self._video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.fps = self._video_cap.get(cv2.CAP_PROP_FPS)
        self._video_info_update_signal.emit(
            VisionSourceInfo(self.video_src_path, self.display_width, self.display_height, self.fps))

    def main(self) -> bool:
        if self._cur_playback_state == EditorPlaybackState.pause:
            self._cv_frame_update_signal.emit(self._last_cap) \
                if self._last_cap is not None \
                else None
            return True
        elif self._cur_playback_state == EditorPlaybackState.stop:
            self._cv_frame_update_signal.emit(VisionCVData(False, None))
            return False

        success, cap = self._video_cap.read()
        if success:
            self._last_cap = VisionCVData(success, cap)
            self._cv_frame_update_signal.emit(self._last_cap)  # vid frame img in numpy format
            if not self.config.using_tracker_vision:
                cap = utils.convert_cv_qt(cv_img=cap, width=self.display_width, height=self.display_height)
                self._frame_update_signal.emit(cap)

        return success

    @pyqtSlot(EditorPlaybackState)
    def on_playback_state_changed_slot(self, new_state):
        self._cur_playback_state = new_state
