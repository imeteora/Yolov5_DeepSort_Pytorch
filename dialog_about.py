from PyQt5.QtWidgets import QDialog

from hf_vision.qt.DialogAbout import Ui_DialogAbout


class DialogAbout(Ui_DialogAbout, QDialog):

    def __init__(self) -> None:
        super().__init__()
        self.setupUi(self)
        self.setModal(True)

    def set_content(self, content: str):
        self.textEditContent.setText(content if content is not None and len(content) else "Hello world")
