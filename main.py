import sys

from PyQt5 import QtWidgets

from MainWindow import MainWindow


def main():
    print("Start")
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    app.exec_()


main()
