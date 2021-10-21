from environment.EnvObject import EnvObject


class Food(EnvObject):
    calories: float

    def __init__(self, x: float, y: float, width: float, calories: float):
        super().__init__(x, y, width)
        self.calories = calories
