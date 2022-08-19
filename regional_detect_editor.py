import os
import sys
from pathlib import Path

import torch

from dialog_about import DialogAbout

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QSizePolicy, QMessageBox, QDialog)

from hf_vision import VisionTrackingProvider, VisionTrackingResult, BoundaryLine, Area
from hf_vision.qt import Ui_EditorMainWindow
from hf_vision.vision_source_provider import VisionSourceProvider, VisionSourceInfo
from vision_tracker_config import VisionTrackingConfig

# boundary lines
boundaryLines = [
    # boundaryLine([655, 450, 1125, 450]),    # for hf video 1
    BoundaryLine([38, 521, 1820, 889]),  # for common test.mp4
    BoundaryLine([755, 194, 1876, 330])
]

# Areas
areas = [
    Area([[804, 334], [529, 482], [1313, 618], [1498, 432]])
]


class EditorMainWindow(Ui_EditorMainWindow, QMainWindow):
    _text = lambda _, raw_str: QtCore.QCoreApplication.translate("EditorMainWindow", raw_str)
    tracking_config = VisionTrackingConfig()

    def __init__(self) -> None:
        super().__init__()
        self.setupUi(self)
        self._setup_canvas()
        self.vision_src_thread = None
        self.vision_tracking_thread = None

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

        self.tracking_config.source = vid_file
        self._create_and_start_detecting_pipeline()

    def onChangeSourceStaticImage(self):
        print(self.onChangeSourceStaticImage.__name__)

    def on_change_yolov5_weights(self):
        print(self.on_change_yolov5_weights.__name__)
        if self._query_check_tracking_running(title='警告', message="视频监测正在运行，是否停止当前监测？"):
            return

        self._release_vision_source_provider_thread()

        yoloV5_weights, _ = QtWidgets.QFileDialog.getOpenFileName(self,
                                                                  self._text('选择YoloV5权重文件'),
                                                                  str(ROOT / 'yolov5/weights'),
                                                                  "*.pt;;*.onnx;;All Files(*)")
        if yoloV5_weights is None:
            return
        self.tracking_config.yolo_weights = yoloV5_weights

    def on_change_reid_ss_weights(self):
        print(self.on_change_reid_ss_weights.__name__)
        if self._query_check_tracking_running('警告', '视频监测正在运行，是否停止当前监测？'):
            return
        self._release_vision_source_provider_thread()
        reid_ss_weights, _ = QtWidgets.QFileDialog.getOpenFileName(self,
                                                                   self._text('选择StrongSort权重文件'),
                                                                   str(ROOT / 'strong_sort/deep/checkpoint'),
                                                                   '*.pth;;All Files(*)')
        if reid_ss_weights is not None:
            self.tracking_config.reid_sort_weights = reid_ss_weights

    def on_change_reid_ss_config(self):
        print(self.on_change_reid_ss_config.__name__)
        if self._query_check_tracking_running('警告', '视频监测正在运行，是否停止当前监测？'):
            return
        self._release_vision_source_provider_thread()
        reid_ss_config, _ = QtWidgets.QFileDialog.getOpenFileName(self,
                                                                  self._text('选择StrongSort配置文件'),
                                                                  str(ROOT / 'strong_sort/configs'),
                                                                  '*.yaml;;All Files(*)')
        if reid_ss_config is not None:
            self.tracking_config.reid_sort_config = reid_ss_config

    def on_draw_regional_boundary_line(self):
        pass

    def on_draw_regional_area(self):
        pass

    def on_canvas_update_notification_received_slot(self, image):
        self.canvas.setPixmap(image)

    def _query_check_tracking_running(self, title: str = '', message: str = '') -> bool:
        if (self.vision_tracking_thread is not None and self.vision_tracking_thread.isRunning) or \
                (self.vision_src_thread is not None and self.vision_src_thread.isRunning):
            return QMessageBox.warning(self, self._text(title),
                                       self._text(message),
                                       buttons=(QMessageBox.Ok, QMessageBox.Cancel),
                                       defaultButton=QMessageBox.Cancel) is QMessageBox.Ok
        return False

    def on_track_and_reid_received_slot(self, tracking_objs: VisionTrackingResult):
        # _str = '\n'.join(list(map(lambda e: e.description, tracking_objs.objects)))
        # print(_str)
        pass

    def on_video_source_info_received_slot(self, info: VisionSourceInfo):
        print(f'Video info: File:{info.source}, frame size:{info.width, info.height}, fps:{info.fps}')
        self.canvas.resize(int(info.width), int(info.height))
        self.updateGeometry()

    def _setup_canvas(self):
        self.canvas = QLabel()
        size_policy = QSizePolicy()
        size_policy.setVerticalPolicy(QSizePolicy.Expanding)
        size_policy.setHorizontalPolicy(QSizePolicy.Expanding)
        self.canvas.setSizePolicy(size_policy)
        self.canvas.setStyleSheet('background : black;')
        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.canvas, 1)
        self.centralwidget.setLayout(vbox)

    def _create_and_start_detecting_pipeline(self):
        self.vision_tracking_thread = VisionTrackingProvider(config=self.tracking_config, boundaryLines=boundaryLines,
                                                             areas=areas)
        self.vision_tracking_thread.connect_cv_pred_signal(self.on_track_and_reid_received_slot)
        self.vision_tracking_thread.connect_tracking_vision_signal(self.on_canvas_update_notification_received_slot)

        self.vision_src_thread = VisionSourceProvider(config=self.tracking_config)
        self.vision_src_thread.set_video_file(self.tracking_config.source)
        self.vision_src_thread.connect_pixel_image_signal(self.on_canvas_update_notification_received_slot)
        self.vision_src_thread.connect_cv_image_signal(self.vision_tracking_thread.on_cv_frame_update_slot)
        self.vision_src_thread.connect_video_info_signal(self.on_video_source_info_received_slot)

        self.vision_src_thread.start()
        self.vision_tracking_thread.start()

    def _release_vision_source_provider_thread(self):
        if self.vision_src_thread is not None and self.vision_src_thread.isRunning():
            self.vision_src_thread.stop()

        if self.vision_tracking_thread is not None and self.vision_tracking_thread.isRunning():
            self.vision_tracking_thread.stop()

        self.vision_src_thread.disconnect_signal() if self.vision_src_thread is not None else None
        self.vision_tracking_thread.disconnect_signal() if self.vision_tracking_thread is not None else None

        self.vision_src_thread = None
        self.vision_tracking_thread = None


if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    window = EditorMainWindow()
    window.show()
    app.exec()
