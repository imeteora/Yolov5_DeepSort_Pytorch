from enum import Enum, Flag, auto, unique
from tkinter import CENTER
from turtle import right
import cv2
from matplotlib.pyplot import text


@unique
class TextAlign(Flag):
    Left = 1 << 0
    Right = 1 << 1
    Center = 1 << 2
    Top = 1 << 3
    Bottom = 1 << 4
    Middle = 1 << 5


def putText(img, string, pos, align: TextAlign = TextAlign.Top | TextAlign.Left, font=cv2.FONT_HERSHEY_PLAIN, font_size=4, text_color=(0, 0, 0), text_thickness=2):
    text_size, _ = cv2.getTextSize(
        str(string), font, font_size, text_thickness)
    x, y = pos[0], pos[1]
    if 0 != (align & TextAlign.Right).value:
        x = int(pos[0] - text_size[0])
    if 0 != (align & TextAlign.Center).value:
        x = int(pos[0] - text_size[0] * 0.5)
    if 0 != (align & TextAlign.Top).value:
        y = int(pos[1] + text_size[1])
    if 0 != (align & TextAlign.Middle).value:
        y = int(pos[1] + text_size[1] * 0.5)
    cv2.putText(img, str(string), (x, y), font,
                font_size, text_color, text_thickness)


# Draw single boundary line
def drawBoundaryLine(img, line):
    x1, y1 = line.p0.x, line.p0.y
    x2, y2 = line.p1.x, line.p1.y
    cv2.line(img, (x1, y1), (x2, y2), line.color, line.line_thickness)

    putText(img, str(line.count1), (x1, y1), TextAlign.Bottom | TextAlign.Left, cv2.FONT_HERSHEY_PLAIN,
            line.font_size, line.text_color, line.text_thickness)
    cv2.drawMarker(img, (x1, y1), line.color, cv2.MARKER_TRIANGLE_UP, 16, 4)

    putText(img, str(line.count2), (x2, y2), TextAlign.Bottom | TextAlign.Right,
            cv2.FONT_HERSHEY_PLAIN, line.font_size, line.text_color, line.text_thickness)
    cv2.drawMarker(img, (x2, y2), line.color, cv2.MARKER_TILTED_CROSS, 16, 4)


# Draw multiple boundary lines
def drawBoundaryLines(img, boundaryLines):
    for line in boundaryLines:
        drawBoundaryLine(img, line)
