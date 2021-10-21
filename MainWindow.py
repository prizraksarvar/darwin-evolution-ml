from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt
import matplotlib

matplotlib.use('Qt5Agg')
DIR = '/home/prizrak/Загрузки/'


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.label = QtWidgets.QLabel()
        canvas = QtGui.QPixmap(400, 300)
        self.label.setPixmap(canvas)
        self.setCentralWidget(self.label)
        self.show()
        # TODO: Start App?
        self.draw()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == Qt.Key_0:
            self.manual_game.handle_action('0')
        if event.key() == Qt.Key_1:
            self.manual_game.handle_action('1')
        # TODO: hook to initializer?

    def draw(self):
        painter = QtGui.QPainter(self.label.pixmap())
        pen = QtGui.QPen()

        pen.setWidth(3)
        pen.setColor(QtGui.QColor("#EB5160"))
        painter.setPen(pen)

        painter.fillRect(0, 0, 400, 300, Qt.black)

        painter.drawRect(50, 50, 100, 100)
        painter.drawRect(60, 60, 150, 100)
        painter.drawRect(70, 70, 100, 150)
        painter.drawRect(80, 80, 150, 100)
        painter.drawRect(90, 90, 100, 150)

        painter.drawLine(10, 10, 300, 200)
        painter.end()
