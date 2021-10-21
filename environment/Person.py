from environment.EnvObject import EnvObject


class Person(EnvObject):
    # Голод
    hunger: float

    def __init__(self, x: float, y: float, width: float, hunger: float):
        super().__init__(x, y, width)
        self.hunger = hunger
