from time import sleep

from PyQt5.QtCore import QObject, pyqtSignal


class Worker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def __init__(self, backgroundLoop):
        super().__init__()
        self.backgroundLoop = backgroundLoop

    def run(self):
        """Long-running task."""
        i = 0
        while True:
            self.backgroundLoop()
            self.progress.emit(i + 1)
        # for i in range(5):
        #     sleep(1)
        #     self.progress.emit(i + 1)
        # self.finished.emit()
