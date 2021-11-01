from math import sin, pi, cos
from random import seed, random

from environment.Environment import Environment
from environment.Food import Food


class LeftRightEnvironment(Environment):
    left_count: int
    right_count: int

    def __init__(self, max_x: float, max_y: float, food_max_count: int, person_max_count: int, food_calories: float,
                 person_hunger: float):

        self.left_count = 0
        self.right_count = 0

        super().__init__(max_x, max_y, food_max_count, person_max_count, food_calories, person_hunger)

    def getNewFood(self):
        person = self.persons[0]
        angle = person.movementAngle
        radius = person.width / 2 + 2
        x2 = radius * sin(pi * 2 * angle / 360) + person.x
        y2 = radius * cos(pi * 2 * angle / 360) + person.y

        ox = 0
        oy = 0

        n = 10
        while n > 0:
            # TODO: сделал просто while ибо так быстрее было чем придумывать формулу расчета координат
            n = n - 1
            point = self.getRandCoords()
            ox = point[0]
            oy = point[1]
            leftSide = ((person.x - x2) * (oy - y2) - (person.y - y2) * (ox - x2)) > 0
            if self.left_count <= self.right_count and leftSide:
                self.left_count = self.left_count + 1
                break
            if self.left_count > self.right_count and not leftSide:
                self.right_count = self.right_count + 1
                break
        return Food(ox, oy, 10, self.food_calories)

    def getRandCoords(self) -> [float, float]:
        return random() * (self.max_x - 60) + 30, random() * (self.max_y - 60) + 30
