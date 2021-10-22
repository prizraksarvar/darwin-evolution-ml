class Control(object):
    moveForward: bool
    moveBack: bool
    rotateLeft: bool
    rotateRight: bool

    def __init__(self):
        self.moveForward = False
        self.moveBack = False
        self.rotateLeft = False
        self.rotateRight = False