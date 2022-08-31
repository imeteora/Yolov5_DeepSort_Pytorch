import os
import sys
from pathlib import Path

# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

for __each_path in list(map(lambda e: str(e), [ROOT, ROOT / 'yolov5', ROOT / 'strong_sort', ROOT / 'hf_vision'])):
    if __each_path not in sys.path:
        sys.path.append(__each_path)

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import torch
from enum import Enum

from hf_vision.core import ThreadStateType
from hf_vision.regional_tracking import Rect

from dialog_about import DialogAbout
from hf_vision.qt.CanvasWidget import CanvasWidget
from hf_vision.qt.mouse_tracker import MouseTracker, MouseEvent

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, pyqtSlot, pyqtSignal, QPoint
from PyQt5.QtWidgets import (QApplication, QMainWindow, QMessageBox, QDialog, QActionGroup)

from hf_vision import VisionTrackingProvider, VisionMOTResult, Area, ThreadState, ObstacleType, Point, \
    SignalSlotCachable, VisionSourceInfo, EditorPlaybackState
from hf_vision.regional_tracking.obstacle_boundary_line import BoundaryLine
from hf_vision.vision_obstacle_edit_object import VisionObstacleEditObject
from hf_vision.qt import Ui_EditorMainWindow
from hf_vision.vision_source_provider import VisionSourceProvider
from vision_tracker_config import VisionTrackingConfig
from regional_detect_editor_private import *

# boundary lines
boundaryLines = [
    # boundaryLine([655, 450, 1125, 450]),    # for hf video 1
    BoundaryLine([[38, 521], [1820, 889]]),  # for common test.mp4
    BoundaryLine([[755, 194], [1876, 330]])
]

# Areas
areas = [
    Area([[804, 334], [529, 482], [1313, 618], [1498, 432]])
]

DEBUG = True


class EditType(Enum):
    normal = 0
    draw_boundary_line = normal + 1
    draw_regional_area = draw_boundary_line + 1


class EditorMainWindow(Ui_EditorMainWindow, QMainWindow, SignalSlotCachable, EditorMainWindowMixin):
    _text = lambda _, raw_str: QtCore.QCoreApplication.translate("EditorMainWindow", raw_str)
    _tracking_config = VisionTrackingConfig()
    _current_edit_state = EditType.normal

    """
    signal for editing obstacle sent to MOT tracking thread, for showing current editing obstacle
    """
    _editing_obstacle_signal = pyqtSignal(VisionObstacleEditObject)
    _editing_obstacle_signal_connection = None

    """
    signal for new confirm edited obstacle append to MOT tracking thread.
    """
    _append_obstacle_signal = pyqtSignal(VisionObstacleEditObject)
    _append_obstacle_signal_connection = None

    _canvas_mouse_clicked_signal = pyqtSignal(MouseEvent)
    _canvas_mouse_clicked_signal_connection = None

    _canvas_mouse_position_changed_signal = pyqtSignal(Point)
    _canvas_mouse_position_changed_signal_connection = None

    _playback_state_change_signal = pyqtSignal(EditorPlaybackState)
    _vid_src_state_changed_signal_connection = None
    _vid_track_state_changed_signal_connection = None
    _tracking_play_state = EditorPlaybackState.stop

    def __init__(self) -> None:
        super().__init__()
        self.setupUi(self)
        self._config_ui()
        self._setup_canvas()
        self._setup_mouse_tracker()
        self.vision_src_thread = None
        self.vision_tracking_thread = None
        self.video_info = None
        self.canvas_rect = Rect.zero()
        self._current_editing_obstacle = VisionObstacleEditObject()
        self._all_obstacles = []

        self._refresh_ui()

    def _config_ui(self):
        self._edit_mode_group = QActionGroup(self)
        self._edit_mode_group.addAction(self.actionSelectNormal)
        self._edit_mode_group.addAction(self.actionDrawLine)
        self._edit_mode_group.addAction(self.actionDrawPolygonArea)
        self.actionSelectNormal.setChecked(True)

    def _refresh_ui(self):
        self.actionStartOrPauseDetect.setIcon(EditorPlaybackState.icon_for_playback_button(self._tracking_play_state))

    def onRegMapNew(self):
        print(self.onRegMapNew.__name__)

    def onRegMapOpen(self):
        print(self.onRegMapOpen.__name__)

    def onRegMapClose(self):
        print(self.onRegMapClose.__name__)

    def onQuitApp(self):
        print(self.onQuitApp.__name__)
        self._release_vision_source_provider_thread()
        quit(0)

    def onAboutApp(self):
        print(self.onAboutApp.__name__)
        _content = \
            f'cuda is available: {torch.cuda.is_available()}\n' + \
            f'{torch.zeros(1).cuda()}\n'
        dialog = DialogAbout()
        dialog.set_content(_content * 100)
        assert (dialog.exec() == QDialog.Accepted)

    def onChangeSourceLocalCamera(self):
        print(self.onChangeSourceLocalCamera.__name__)

    def onChangeSourceRemoteCamera(self):
        print(self.onChangeSourceRemoteCamera.__name__)

    def onChangeSourceMediaVideo(self):
        print(self.onChangeSourceMediaVideo.__name__)
        self._release_vision_source_provider_thread()

        vid_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "打开视频文件", os.getcwd(), "*.mp4;;*.mov;;All Files(*)")
        if vid_file is None:
            return

        self._tracking_config.source = vid_file

    def onChangeSourceStaticImage(self):
        print(self.onChangeSourceStaticImage.__name__)

    def on_change_yolov5_weights(self):
        print(self.on_change_yolov5_weights.__name__)
        if not self._query_check_tracking_running(title='警告', message="视频监测正在运行，是否停止当前监测？"):
            return

        self._release_vision_source_provider_thread()

        yoloV5_weights, _ = QtWidgets.QFileDialog.getOpenFileName(self,
                                                                  self._text('选择YoloV5权重文件'),
                                                                  str(ROOT / 'yolov5/weights'),
                                                                  "*.pt;;*.onnx;;All Files(*)")
        if yoloV5_weights is None:
            return
        self._tracking_config.yolo_weights = yoloV5_weights

    def on_change_reid_ss_weights(self):
        print(self.on_change_reid_ss_weights.__name__)
        if not self._query_check_tracking_running('警告', '视频监测正在运行，是否停止当前监测？'):
            return
        self._release_vision_source_provider_thread()
        reid_ss_weights, _ = QtWidgets.QFileDialog.getOpenFileName(self,
                                                                   self._text('选择StrongSort权重文件'),
                                                                   str(ROOT / 'strong_sort/deep/checkpoint'),
                                                                   '*.pth;;All Files(*)')
        if reid_ss_weights is not None:
            self._tracking_config.reid_sort_weights = reid_ss_weights

    def on_change_reid_ss_config(self):
        print(self.on_change_reid_ss_config.__name__)
        if not self._query_check_tracking_running('警告', '视频监测正在运行，是否停止当前监测？'):
            return
        self._release_vision_source_provider_thread()
        reid_ss_config, _ = QtWidgets.QFileDialog.getOpenFileName(self,
                                                                  self._text('选择StrongSort配置文件'),
                                                                  str(ROOT / 'strong_sort/configs'),
                                                                  '*.yaml;;All Files(*)')
        if reid_ss_config is not None:
            self._tracking_config.reid_sort_config = reid_ss_config

    def on_draw_reset_normal(self):
        print(self.on_draw_reset_normal.__name__)
        self._change_edit_state(new_state=EditType.normal)

    def on_draw_regional_boundary_line(self):
        print(self.on_draw_regional_boundary_line.__name__)
        self._change_edit_state(new_state=EditType.draw_boundary_line)

    def on_draw_regional_area(self):
        print(self.on_draw_regional_area.__name__)
        self._change_edit_state(new_state=EditType.draw_regional_area)

    def on_tracking_start_or_pause(self):
        if not self._tracking_config.is_ready:
            return

        # src thread and MOT thread, etc. All starts
        if self._tracking_play_state == EditorPlaybackState.stop:
            self._create_and_start_detecting_pipeline()

        # change the current playback state and refresh UI
        self._tracking_play_state = EditorPlaybackState.start \
            if self._tracking_play_state in [EditorPlaybackState.stop, EditorPlaybackState.pause] \
            else EditorPlaybackState.pause

        self._evt_notify_source_tracking_with_playback_state()
        self._refresh_ui()

    def on_tracking_stop(self):
        # stop all threads, include src thread and MOT thread.
        if self._tracking_play_state in [EditorPlaybackState.pause, EditorPlaybackState.start]:
            self._release_vision_source_provider_thread()

        # change current playback state and refresh ui
        self._tracking_play_state = EditorPlaybackState.stop
        self._evt_notify_source_tracking_with_playback_state()
        self._refresh_ui()

    def _evt_notify_source_tracking_with_playback_state(self):
        self._playback_state_change_signal.emit(self._tracking_play_state)

    def on_canvas_update_notification_received_slot(self, image):
        if image is not None:
            self.canvas.set_pixmap(image.scaled(self.canvas.width(),
                                                self.canvas.height(),
                                                Qt.KeepAspectRatio,
                                                Qt.SmoothTransformation))
        else:
            self.canvas.set_pixmap(None)

    def _evt_update_cursor(self):
        pass

    def _query_check_tracking_running(self, title: str = '', message: str = '') -> bool:
        if (self.vision_tracking_thread is not None and self.vision_tracking_thread.isRunning) or \
                (self.vision_src_thread is not None and self.vision_src_thread.isRunning):
            return QMessageBox.warning(self, self._text(title),
                                       self._text(message),
                                       buttons=QMessageBox.Ok | QMessageBox.Cancel,
                                       defaultButton=QMessageBox.Cancel) is QMessageBox.Ok
        return True

    def on_track_and_reid_received_slot(self, tracking_objs: VisionMOTResult):
        # _str = '\n'.join(list(map(lambda e: e.description, tracking_objs.objects)))
        # print(_str)
        pass

    def on_video_source_info_received_slot(self, info: VisionSourceInfo):
        print(f'Video info: File:{info.source}, frame size:{info.width, info.height}, fps:{info.fps}')
        self.video_info = info
        self.updateGeometry()

    @pyqtSlot(QtCore.QPoint)
    def _on_canvas_mouse_move_evt_slot(self, pos):
        _scaled_pos = self._mouse_scaled_pt(pos)
        self._canvas_mouse_position_changed_signal.emit(_scaled_pos)

        self.statusbar.showMessage(f'{pos.x(), pos.y()}')

    @pyqtSlot(MouseEvent)
    def _on_canvas_mouse_click_evt_slot(self, evt):
        if type(evt) is not MouseEvent:
            return

        __kb_modifiers = QApplication.queryKeyboardModifiers()
        __is_ctrl = __kb_modifiers == Qt.ControlModifier
        __is_alt = __kb_modifiers == Qt.AltModifier
        __is_shift = __kb_modifiers == Qt.ShiftModifier

        evt.pos = self._mouse_scaled_pt(evt.pos)

        if self._current_edit_state in [EditType.draw_boundary_line, EditType.draw_regional_area]:
            if evt.is_left_release:
                self._current_editing_obstacle.push(evt.pos)
                if __is_ctrl:
                    self._current_editing_obstacle.is_done = True
            elif evt.is_right_release:
                self._current_editing_obstacle.pop()
            elif evt.is_mid_release:
                # cancel current obstacle editing and re-new one
                self._handle_new_edit_state(old_state=self._current_edit_state, new_state=self._current_edit_state)

            if self._current_editing_obstacle.is_done:
                self._append_obstacle_signal.emit(self._current_editing_obstacle)
                self._current_editing_obstacle = VisionObstacleEditObject.none()
                # append this obstacle to the vision tracker's object list.
                self._handle_new_edit_state(old_state=self._current_edit_state, new_state=self._current_edit_state)

        elif self._current_edit_state == EditType.normal:
            self._canvas_mouse_clicked_signal.emit(evt)

    @pyqtSlot(ThreadState)
    def _on_vision_src_thread_state_update_slot(self, thread_info):
        if thread_info.state == ThreadStateType.stop:
            self._release_vision_source_provider_thread()
        else:
            pass

    def _mouse_scaled_pt(self, pos: QPoint) -> Point:
        diff_width = 0.5 * (self.canvas.image_rect.width() - self.canvas.width()) \
            if self.canvas.image_rect is not None \
            else 0.0
        diff_height = 0.5 * (self.canvas.image_rect.height() - self.canvas.height()) \
            if self.canvas.image_rect is not None \
            else 0.0

        _ratio = 0.0
        if self.video_info is not None and self.canvas.image_rect is not None:
            _ratio = float(self.video_info.width) / float(self.canvas.image_rect.width())
            _ratio = max(float(self.video_info.height / float(self.canvas.image_rect.height())), _ratio)
            _ratio = max(0.0, _ratio)

        _result = Point(pos.x() + diff_width, pos.y() + diff_height)

        return _result \
            if _ratio == 0.0 \
            else _result.scale(_ratio)

    def _change_edit_state(self, new_state: EditType):
        if new_state == self._current_edit_state:
            return
        self._handle_old_edit_state(old_state=self._current_edit_state, new_state=new_state)
        self._current_edit_state = new_state
        self._handle_new_edit_state(old_state=self._current_edit_state, new_state=new_state)
        self._evt_update_cursor()

    def _handle_old_edit_state(self, old_state: EditType, new_state: EditType):
        if old_state in [EditType.normal, EditType.draw_boundary_line, EditType.draw_regional_area]:
            self._current_editing_obstacle = VisionObstacleEditObject.none()
            self._editing_obstacle_signal.emit(self._current_editing_obstacle)

    def _handle_new_edit_state(self, old_state: EditType, new_state: EditType):
        if new_state == EditType.draw_boundary_line:
            self._current_editing_obstacle = VisionObstacleEditObject(type=ObstacleType.boundary_line)
            self._editing_obstacle_signal.emit(self._current_editing_obstacle)
        elif new_state == EditType.draw_regional_area:
            self._current_editing_obstacle = VisionObstacleEditObject(type=ObstacleType.area)
            self._editing_obstacle_signal.emit(self._current_editing_obstacle)
        elif new_state == EditType.normal:
            self._current_editing_obstacle = VisionObstacleEditObject.none()
            self._editing_obstacle_signal.emit(self._current_editing_obstacle)
        else:
            pass

    def _setup_canvas(self):
        self.canvas = CanvasWidget()
        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.canvas, 1)
        vbox.setContentsMargins(0, 0, 0, 0)
        self.centralwidget.setLayout(vbox)

    def _setup_mouse_tracker(self):
        if self.canvas is None:
            return

        self._canvas_mouse_tracker = MouseTracker(widget=self.canvas, enable=True)
        self._canvas_mouse_tracker.position_change_event_signal.connect(self._on_canvas_mouse_move_evt_slot)
        self._canvas_mouse_tracker.click_event_signal.connect(self._on_canvas_mouse_click_evt_slot)

    def _create_and_start_detecting_pipeline(self):
        self.vision_tracking_thread = VisionTrackingProvider(config=self._tracking_config, boundary_lines=boundaryLines,
                                                             areas=areas)
        self.vision_tracking_thread.connect_cv_pred_signal(self.on_track_and_reid_received_slot)
        self.vision_tracking_thread.connect_tracking_vision_signal(self.on_canvas_update_notification_received_slot)

        self.vision_src_thread = VisionSourceProvider(config=self._tracking_config)
        self.vision_src_thread.connect_thread_state_update_signal(self._on_vision_src_thread_state_update_slot)
        # self.vision_src_thread.connect_pixel_image_signal(self.on_canvas_update_notification_received_slot)
        self.vision_src_thread.connect_cv_image_signal(self.vision_tracking_thread.on_cv_frame_update_slot)
        self.vision_src_thread.connect_video_info_signal(self.on_video_source_info_received_slot)
        self.vision_src_thread.set_video_file(self._tracking_config.source)

        self._canvas_mouse_position_changed_signal_connection = \
            self._canvas_mouse_position_changed_signal.connect(
                self.vision_tracking_thread.on_mouse_position_changed_slot)

        self._canvas_mouse_clicked_signal_connection = \
            self._canvas_mouse_clicked_signal.connect(self.vision_tracking_thread.on_mouse_clicked_slot)

        self._editing_obstacle_signal_connection = \
            self._editing_obstacle_signal.connect(self.vision_tracking_thread.on_edit_obstacle_arrived_slot)

        self._append_obstacle_signal_connection = \
            self._append_obstacle_signal.connect(self.vision_tracking_thread.on_append_obstacle_slot)

        self._vid_src_state_changed_signal_connection = \
            self._playback_state_change_signal.connect(self.vision_src_thread.on_playback_state_changed_slot)

        self._vid_track_state_changed_signal_connection = \
            self._playback_state_change_signal.connect(self.vision_tracking_thread.on_playback_state_changed_slot)

        self.vision_src_thread.start()
        self.vision_tracking_thread.start()

    def _release_vision_source_provider_thread(self):
        if self.vision_src_thread is not None and self.vision_src_thread.isRunning():
            self.vision_src_thread.stop()

        if self.vision_tracking_thread is not None and self.vision_tracking_thread.isRunning():
            self.vision_tracking_thread.stop()

        if self._canvas_mouse_position_changed_signal_connection is not None:
            self._canvas_mouse_position_changed_signal.disconnect(self._canvas_mouse_position_changed_signal_connection)
            self._canvas_mouse_position_changed_signal_connection = None

        if self._canvas_mouse_clicked_signal_connection is not None:
            self._canvas_mouse_clicked_signal.disconnect(self._canvas_mouse_clicked_signal_connection)
            self._canvas_mouse_clicked_signal_connection = None

        if self._editing_obstacle_signal_connection is not None:
            self._editing_obstacle_signal.disconnect(self._editing_obstacle_signal_connection)
            self._editing_obstacle_signal_connection = None

        if self._append_obstacle_signal_connection is not None:
            self._append_obstacle_signal.disconnect(self._append_obstacle_signal_connection)
            self._append_obstacle_signal_connection = None

        if self._vid_src_state_changed_signal_connection is not None:
            self._playback_state_change_signal.disconnect(self._vid_src_state_changed_signal_connection)
            self._vid_src_state_changed_signal_connection = None

        if self._vid_track_state_changed_signal_connection is not None:
            self._playback_state_change_signal.disconnect(self._vid_track_state_changed_signal_connection)
            self._vid_track_state_changed_signal_connection = None

        self.vision_tracking_thread.disconnect_signal() if self.vision_tracking_thread is not None else None
        self.vision_src_thread.disconnect_signal() if self.vision_src_thread is not None else None

        self.vision_src_thread = None
        self.vision_tracking_thread = None


if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    window = EditorMainWindow()
    window.show()
    app.exec()
