from PyQt5.QtCore import QSize, QRect
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QPixmap
from PyQt5.QtWidgets import QWidget, QStyleOption, QStyle


class CanvasWidget(QWidget):
    pixmap = None
    _sizeHint = QSize()
    ratio = Qt.KeepAspectRatio
    transformation = Qt.SmoothTransformation

    def __init__(self, pixmap=None):
        super().__init__()
        self.scaled = None
        self.set_pixmap(pixmap)
        self.setStyleSheet("""
        QWidget {
            border: 0px;
            background: black;
        }
        """)

    def set_pixmap(self, pixmap):
        if self.pixmap != pixmap:
            self.pixmap = pixmap
            if isinstance(pixmap, QPixmap):
                self._sizeHint = pixmap.size()
            else:
                self._sizeHint = QSize()
            self.updateGeometry()
            self.update_scaled()

    def set_aspect_ratio(self, ratio):
        if self.ratio != ratio:
            self.ratio = ratio
            self.update_scaled()

    def set_transformation(self, transformation):
        if self.transformation != transformation:
            self.transformation = transformation
            self.update_scaled()

    def update_scaled(self):
        if self.pixmap:
            self.scaled = self.pixmap.scaled(self.size(), self.ratio, self.transformation)
        self.update()

    def sizeHint(self):
        return self._sizeHint

    @property
    def image_rect(self) -> QRect:
        return self.scaled.rect() \
            if self.scaled is not None \
            else None

    def resizeEvent(self, event):
        self.update_scaled()

    def paintEvent(self, event):
        opt = QStyleOption()
        opt.initFrom(self)
        bg_painter = QPainter(self)
        self.style().drawPrimitive(QStyle.PE_Widget, opt, bg_painter, self)

        if not self.pixmap:
            QWidget.paintEvent(self, event)
            return

        r = self.scaled.rect()
        r.moveCenter(self.rect().center())
        bg_painter.drawPixmap(r, self.scaled)

        QWidget.paintEvent(self, event)
