import sys

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt
from MainWindow import MainWindow
from environment.Environment import Environment
from environment.control import Control
from game_score import GameScore
from ml.learn_spinal_cord import SpinalCordLearner


class Application(object):
    app: QtWidgets.QApplication
    window: MainWindow

    environment: Environment
    controls: [Control]
    scores: [GameScore]
    externalControl: bool

    def __init__(self, environment: Environment, controls: [Control], scores: [GameScore], loopFun, restartedFun, externalControl: bool):
        print("Start")

        self.environment = environment
        self.controls = controls
        self.scores = scores
        self.externalControl = externalControl
        self.loopFun = loopFun
        self.restartedFun = restartedFun

        self.app = QtWidgets.QApplication(sys.argv)
        self.window = MainWindow(self.loop, self.backgroundLoop, self.keyPressEventHook, self.keyReleaseEventHook)
        self.app.exec_()

    def loop(self):
        if not self.externalControl:
            self.environment.tickUpdate(self.controls)
        self.window.draw(self.environment)
        pass

    def backgroundLoop(self):
        if not self.externalControl:
            return
        self.loopFun()
        self.environment.tickUpdate(self.controls)
        if self.environment.persons[0].hunger == 100:
            self.environment.reinit()
            if self.restartedFun is not None:
                self.restartedFun()
        if self.environment.persons[0].hunger == 0:
            # exit(0)
            self.app.quitOnLastWindowClosed()
            self.app.closeAllWindows()

    def clearControl(self):
        self.controls[0].moveForward = False
        self.controls[0].moveBack = False
        self.controls[0].rotateLeft = False
        self.controls[0].rotateRight = False

    def keyPressEventHook(self, event: QtGui.QKeyEvent):
        if self.externalControl:
            return
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
        if self.externalControl:
            return
        if event.key() == Qt.Key_Up:
            self.controls[0].moveForward = False
        if event.key() == Qt.Key_Down:
            self.controls[0].moveBack = False
        if event.key() == Qt.Key_Left:
            self.controls[0].rotateLeft = False
        if event.key() == Qt.Key_Right:
            self.controls[0].rotateRight = False
        pass


environment = Environment(400, 300, 1, 1, 30)
controls = [Control()]
scores = [GameScore()]

# Ручное управление
# app = Application(environment, controls, None, None, False)

# ML с обучением
learner = SpinalCordLearner(environment, controls, scores)
app = Application(environment, controls, scores, learner.learnLoop, learner.gameRestarted, True)
# app = Application(environment, controls, learner.testLoop, learner.gameRestarted, True)
learner.done()
