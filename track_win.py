import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal, Qt, QThread
from PyQt5.QtGui import QPixmap
from PyQt5.QtMultimedia import QCameraInfo
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QDesktopWidget, QStatusBar

import track

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    image = None

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        # capture from web cam
        self._run_flag = True
        self.cap = cv2.VideoCapture(0)
        while self._run_flag:
            # self.change_pixmap_signal.emit(self.image) if self.image is not None else None
            ret, cv_img = self.cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)
        # shut down capture system
        self.cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False


class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.available_cameras = QCameraInfo.availableCameras()  # Getting available cameras

        cent = QDesktopWidget().availableGeometry().center()  # Finds the center of the screen
        self.setStyleSheet("background-color: white;")
        self.resize(1400, 800)
        self.frameGeometry().moveCenter(cent)
        self.setWindowTitle('S L A I T')
        self.initWindow()

    ########################################################################################################################
    #                                                   Windows                                                            #
    ########################################################################################################################
    def initWindow(self):
        # create the video capture thread
        self.thread = VideoThread()

        # Label with the name of the co-founders
        self.label = QtWidgets.QLabel(self)  # Create label
        self.label.setText('By Evgeny Fomin and Antonio Domènech')  # Add text to label
        self.label.move(30, 550)  # Allocate label in window
        self.label.resize(300, 20)  # Set size for the label
        self.label.setAlignment(Qt.AlignCenter)  # Align text in the label

        # Button to start video
        self.ss_video = QtWidgets.QPushButton(self)
        self.ss_video.setText('Start video')
        self.ss_video.move(769, 100)
        self.ss_video.resize(300, 100)
        self.ss_video.clicked.connect(self.ClickStartVideo)

        # Status bar
        self.status = QStatusBar()
        self.status.setStyleSheet("background : lightblue;")  # Setting style sheet to the status bar
        self.setStatusBar(self.status)  # Adding status bar to the main window
        self.status.showMessage('Ready to start')

        self.image_label = QLabel(self)
        self.disply_width = 669
        self.display_height = 501
        self.image_label.resize(self.disply_width, self.display_height)
        self.image_label.setStyleSheet("background : black;")
        self.image_label.move(0, 0)

    ########################################################################################################################
    #                                                   Buttons                                                            #
    ########################################################################################################################
    # Activates when Start/Stop video button is clicked to Start (ss_video
    def ClickStartVideo(self):
        # Change label color to light blue
        self.ss_video.clicked.disconnect(self.ClickStartVideo)
        self.status.showMessage('Video Running...')
        # Change button to stop
        self.ss_video.setText('Stop video')
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)

        # start the thread
        self.thread.start()
        self.ss_video.clicked.connect(self.thread.stop)  # Stop the video if button clicked
        self.ss_video.clicked.connect(self.ClickStopVideo)

        # track.run(source=ROOT / '../../../test.mp4',
        #           yolo_weights=ROOT / 'yolov5/weights/yolov5s6.pt',
        #           strong_sort_weights=ROOT / 'strong_sort/deep/checkpoint/osnet_x0_25_msmt17.pth',
        #           dnn=True,
        #           half=True,
        #           hide_conf=True,
        #           hide_class=True,
        #           classes=0,
        #           imgsz=512,
        #           video_thread=self.thread)

    # Activates when Start/Stop video button is clicked to Stop (ss_video)
    def ClickStopVideo(self):
        self.thread.change_pixmap_signal.disconnect()
        self.ss_video.setText('Start video')
        self.status.showMessage('Ready to start')
        self.ss_video.clicked.disconnect(self.ClickStopVideo)
        self.ss_video.clicked.disconnect(self.thread.stop)
        self.ss_video.clicked.connect(self.ClickStartVideo)

    ########################################################################################################################
    #                                                   Actions                                                            #
    ########################################################################################################################

    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        # p = convert_to_Qt_format.scaled(801, 801, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MyWindow()
    win.show()
    sys.exit(app.exec())
