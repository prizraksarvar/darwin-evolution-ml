from math import sin, cos, pi

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt, QTimer, QThread
import matplotlib

from Worker import Worker
from environment.Environment import Environment

matplotlib.use('Qt5Agg')
DIR = '/home/prizrak/Загрузки/'


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, loop, backgroundLoop, keyPressEventHook, keyReleaseEventHook, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.extLoop = loop
        self.backgroundLoop = backgroundLoop
        self.keyPressEventHook = keyPressEventHook
        self.keyReleaseEventHook = keyReleaseEventHook
        self.label = QtWidgets.QLabel()
        canvas = QtGui.QPixmap(400, 300)
        self.label.setPixmap(canvas)
        self.setCentralWidget(self.label)
        self.show()

        self.timer = QTimer()
        self.timer.timeout.connect(self.loop)
        # 60 кадров в секунду
        self.timer.start(1000 / 60)

        self.draw()
        # Фоновый воркер что то я подумал заморочки с мьютексами это через чур пока
        # self.runWorkerTask()

        if self.backgroundLoop is not None:
            self.timerBackground = QTimer()
            self.timerBackground.timeout.connect(self.backgroundLoop)
            self.timerBackground.start(0)


    def loop(self):
        self.extLoop()

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
        super().keyReleaseEvent(event)
        self.keyReleaseEventHook(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == Qt.Key_0:
            pass
        if event.key() == Qt.Key_1:
            pass
        self.keyPressEventHook(event)

    def draw(self, environment: Environment = None):
        painter = QtGui.QPainter(self.label.pixmap())
        # Очищаем
        painter.fillRect(0, 0, 400, 300, Qt.black)

        penFood = QtGui.QPen()
        penFood.setWidth(3)
        penFood.setColor(QtGui.QColor("#EB5160"))

        penPerson = QtGui.QPen()
        penPerson.setWidth(3)
        penPerson.setColor(QtGui.QColor("#00EB00"))

        if environment is None:
            painter.end()
            return

        painter.setPen(penFood)
        for food in environment.foods:
            painter.drawEllipse(food.x - food.width / 2, food.y - food.width / 2, food.width, food.width)

        painter.setPen(penPerson)
        painter.drawText(5, 15, "Hunger")
        i = 0
        for person in environment.persons:
            painter.drawEllipse(person.x - person.width / 2, person.y - person.width / 2, person.width, person.width)
            angle = person.movementAngle
            radius = person.width / 2 + 2
            x = radius * sin(pi * 2 * angle / 360) + person.x
            y = radius * cos(pi * 2 * angle / 360) + person.y

            x2 = person.x
            y2 = person.y
            painter.drawLine(x, y, x2, y2)

            painter.drawText(5, 25 + 10 * i, "Person{0}: {1:.2f}%".format(i + 1, person.hunger))
            i = i + 1

        # painter.drawRect(60, 60, 150, 100)

        painter.end()

        self.repaint()

    def runWorkerTask(self):
        if self.backgroundLoop is None:
            return
        # Step 2: Create a QThread object
        self.thread = QThread()
        # Step 3: Create a worker object
        self.worker = Worker(self.backgroundLoop)
        # Step 4: Move worker to the thread
        self.worker.moveToThread(self.thread)
        # Step 5: Connect signals and slots
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.progress.connect(self.reportProgress)
        # Step 6: Start the thread
        self.thread.start()

        # Final resets
        # self.thread.finished.connect(
        #     lambda: self.longRunningBtn.setEnabled(True)
        # )
        # self.thread.finished.connect(
        #     lambda: self.stepLabel.setText("Long-Running Step: 0")
        # )
