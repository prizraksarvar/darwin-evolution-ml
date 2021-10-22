import sys

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt
from MainWindow import MainWindow
from environment.Environment import Environment
from environment.control import Control


class Application(object):
    app: QtWidgets.QApplication
    window: MainWindow

    environment: Environment
    controls: [Control]

    def __init__(self):
        print("Start")

        self.environment = Environment(400, 300, 1, 1)
        self.controls = [Control()]

        self.app = QtWidgets.QApplication(sys.argv)
        self.window = MainWindow(self.loop, self.keyPressEventHook, self.keyReleaseEventHook)
        self.app.exec_()

    def loop(self):
        self.environment.tickUpdate(self.controls)
        self.window.draw(self.environment)
        # TODO: здесь может не очищатся значение, нужно проверить
        pass

    def clearControl(self):
        self.controls[0].moveForward = False
        self.controls[0].moveBack = False
        self.controls[0].rotateLeft = False
        self.controls[0].rotateRight = False

    def keyPressEventHook(self, event: QtGui.QKeyEvent):
        if event.key() == Qt.Key_Up:
            self.controls[0].moveForward = True
        if event.key() == Qt.Key_Down:
            self.controls[0].moveBack = True
        if event.key() == Qt.Key_Left:
            self.controls[0].rotateLeft = True
        if event.key() == Qt.Key_Right:
            self.controls[0].rotateRight = True
        pass

    def keyReleaseEventHook(self, event: QtGui.QKeyEvent):
        if event.key() == Qt.Key_Up:
            self.controls[0].moveForward = False
        if event.key() == Qt.Key_Down:
            self.controls[0].moveBack = False
        if event.key() == Qt.Key_Left:
            self.controls[0].rotateLeft = False
        if event.key() == Qt.Key_Right:
            self.controls[0].rotateRight = False
        pass


app = Application()
