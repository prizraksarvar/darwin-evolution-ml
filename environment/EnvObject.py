class EnvObject(object):
    x: float
    y: float
    width: float
    movementAngle: float
    movementSpeed: float

    def __init__(self, x: float, y: float, width: float):
        self.x = x
        self.y = y
        self.width = width
        self.movementAngle = 0
        self.movementSpeed = 0

