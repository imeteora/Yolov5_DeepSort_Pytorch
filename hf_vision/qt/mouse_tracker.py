from PyQt5 import QtCore
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import QWidget


class MouseEvent(object):
    def __init__(self, button, pos, event_type: int = 0):
        super().__init__()
        self.type = event_type
        self.button = button
        self.pos = pos

    @property
    def is_left_button(self) -> bool:
        return self.button == Qt.LeftButton

    @property
    def is_right_button(self) -> bool:
        return self.button == Qt.RightButton

    @property
    def is_mid_button(self) -> bool:
        return self.button == Qt.MidButton

    @property
    def is_press(self) -> bool:
        return self.type == QMouseEvent.MouseButtonPress

    @property
    def is_release(self) -> bool:
        return self.type == QMouseEvent.MouseButtonRelease

    @property
    def is_left_press(self) -> bool:
        return self.is_left_button and self.is_press

    @property
    def is_mid_press(self) -> bool:
        return self.is_mid_button and self.is_press

    @property
    def is_right_press(self) -> bool:
        return self.is_right_button and self.is_press

    @property
    def is_left_release(self) -> bool:
        return self.is_left_button and self.is_release

    @property
    def is_mid_release(self) -> bool:
        return self.is_mid_button and self.is_release

    @property
    def is_right_release(self) -> bool:
        return self.is_right_button and self.is_release

    @property
    def raw_pos(self) -> [int]:
        return [self.pos.x(), self.pos.y()]


class MouseTracker(QtCore.QObject):
    position_change_event_signal = pyqtSignal(QtCore.QPoint)
    click_event_signal = pyqtSignal(MouseEvent)

    def __init__(self, widget: QWidget, enable: bool = True):
        super().__init__(widget)
        self._widget = widget

        self.enable = enable
        self.widget.installEventFilter(self)

    @property
    def enable(self) -> bool:
        return self.widget.hasMouseTracking()

    @enable.setter
    def enable(self, new_value: bool):
        self.widget.setMouseTracking(new_value)

    @property
    def widget(self):
        return self._widget

    def eventFilter(self, target, evt):
        if target is None or target is not self.widget or type(evt) is not QMouseEvent:
            return super().eventFilter(target, evt)

        mouse_evt_type = evt.type()
        mouse_btn_type = evt.button()

        if mouse_evt_type == QtCore.QEvent.MouseMove:
            self.position_change_event_signal.emit(evt.pos())
        elif mouse_evt_type in [QMouseEvent.MouseButtonPress, QMouseEvent.MouseButtonRelease]:
            self.click_event_signal.emit(MouseEvent(button=evt.button(), pos=evt.pos(), event_type=evt.type())) \
                if mouse_btn_type in [Qt.LeftButton, Qt.RightButton, Qt.MidButton] \
                else None
        else:
            pass

        return super().eventFilter(target, evt)
