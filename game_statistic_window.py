import time

import hiddenlayer as hl
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt, QTimer, QThread
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from environment.Environment import Environment

matplotlib.use('Qt5Agg')


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=5, dpi=100):
        # fig = Figure(figsize=(width, height), dpi=dpi)
        fig = Figure(figsize=(width, height))
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class GameStatisticWindow(QtWidgets.QMainWindow):
    def __init__(self, app, *args, **kwargs):
        super(GameStatisticWindow, self).__init__(*args, **kwargs)

        self.app = app
        sc = MplCanvas(self, width=10, height=10, dpi=100)
        self.canvas = sc
        self.setCentralWidget(sc)

        self.canvas.figure.clf()
        self.canvas.axes = self.canvas.figure.add_subplot(111)

        self.canvas.axes.set_xlim(xmax=2000, xmin=0)
        self.canvas.axes.set_ylim(ymax=200, ymin=0)

        self.plot_rf_food = self.canvas.axes.plot([], [], color='blue', label="Got foods count")[0]
        self.plot_rf_loss = self.canvas.axes.plot([], [], color='red', label="Loss")[0]
        self.canvas.axes.legend()

        self.start_time = time.time()
        self.draw_count = 0

        self.x_score_list = []
        self.x_loss_list = []
        self.y_list = []

        self.xmax = 1
        self.ymax = 1
        self.xmin = 0
        self.ymin = 0

        self.show()
        self.draw(0, 0)

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.save()
        self.app.quitOnLastWindowClosed()
        self.app.closeAllWindows()

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
        super().keyReleaseEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        super().keyPressEvent(event)
        if event.key() == Qt.Key_0:
            pass
        if event.key() == Qt.Key_Minus:
            pass

    def draw(self, score: int, loss: float):
        self.draw_count = self.draw_count + 1

        self.y_list.append(self.draw_count)
        self.x_score_list.append(score)
        self.x_loss_list.append(loss)

        if self.ymax < score:
            self.ymax = score
        if self.ymax < int(loss):
            self.ymax = int(loss)
        if self.ymin > score:
            self.ymin = score
        if self.ymin > int(loss):
            self.ymin = int(loss)

        self.xmax = len(self.y_list) + 5

        self.canvas.axes.set_xlim(xmax=self.xmax, xmin=self.xmin)
        self.canvas.axes.set_ylim(ymax=self.ymax, ymin=self.ymin)

        self.plot_rf_food.set_data(self.y_list, self.x_score_list)
        self.plot_rf_loss.set_data(self.y_list, self.x_loss_list)
        exec_time = round(time.time() - self.start_time, 2)
        self.canvas.axes.set_title(f"Игра № {self.draw_count}\n--- {exec_time} seconds ---", fontsize=12)

        self.canvas.draw()
        self.repaint()

    def save(self):
        self.canvas.figure.savefig('game_statistic.png')
