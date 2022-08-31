from enum import Enum

from PyQt5 import QtGui
from PyQt5.QtGui import QIcon


class EditorPlaybackState(Enum):
    stop = 0
    start = stop + 1
    pause = start + 1

    @classmethod
    def icon_for_playback_button(self, type) -> QIcon:
        pixmap = None
        if type == EditorPlaybackState.start:
            pixmap = QtGui.QPixmap(":/icon/icons/暂停.png")
        elif type == EditorPlaybackState.pause or type == EditorPlaybackState.stop:
            pixmap = QtGui.QPixmap(":/icon/icons/播放2.png")
        else:
            pass
        icon = QIcon()
        icon.addPixmap(pixmap, QIcon.Normal, QIcon.Off)
        return icon
