from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPaintEvent, QPainter, QColor, QFont, QPixmap, QBrush
from PyQt5.QtWidgets import QWidget


class WorkspaceWidget(QWidget):
    qp = QPainter()

    def __init__(self, parent) -> None:
        super().__init__(parent=parent)

    def paintEvent(self, evt: QPaintEvent):
        self.qp.begin(self)
        self.qp.setPen(QColor(Qt.red))
        self.qp.setFont(QFont('Arial', 20))
        self.qp.drawText(10, 50, "hello Python")
        self.qp.setPen(QColor(Qt.blue))
        self.qp.drawLine(10, 100, 100, 100)
        self.qp.drawRect(10, 150, 150, 100)
        self.qp.setPen(QColor(Qt.yellow))
        self.qp.drawEllipse(100, 50, 100, 50)
        self.qp.drawPixmap(220, 10, QPixmap("pythonlogo.png"))
        self.qp.fillRect(20, 175, 130, 70, QBrush(Qt.SolidPattern))
        self.qp.end()

        # painter = QPainter()
        #
        # painter.begin(self)
        # # painter.setPen(QColor(255, 0, 255, 255))
        # # painter.drawLine(0, 0, 200, 200)
        #
        # painter.setPen(QColor(0xff00ffff))
        # painter.setFont(QFont('Arial', 20))
        # painter.drawText(0, 0,  'Hello world')
        #
        # painter.end()
