from math import sin, pi, cos
from random import seed, random

from environment.EnvObject import EnvObject
from environment.Food import Food
from environment.Person import Person
from environment.calc import cycle_interception
from environment.control import Control


class Environment(object):
    foods: [Food]
    persons: [Person]
    food_max_count: int
    person_max_count: int
    max_x: float
    max_y: float

    def __init__(self, max_x: float, max_y: float, food_max_count: int, person_max_count: int):
        self.max_x = max_x
        self.max_y = max_y
        self.food_max_count = food_max_count
        self.person_max_count = person_max_count
        self.foods = [Food(0, 0, 0, 0)]*self.food_max_count
        self.persons = [Person(0, 0, 0, 0)]*self.person_max_count

        # Инициализируем генератор случайных чисел
        seed(version=2)
        self.reinit()

    def reinit(self):
        for i in range(0, self.food_max_count):
            point = self.getRandCoords()
            self.foods[i] = Food(point[0], point[1], 10, 30)

        for i in range(0, self.person_max_count):
            point = self.getRandCoords()
            self.persons[i] = Person(point[0], point[1], 10, 10)

    def tickUpdate(self, controls: [Control]):
        for i in range(0, self.food_max_count):
            self.moveObject(self.foods[i])
            self.moveResistanceProcess(self.foods[i])
            self.wallCollisitionProcess(self.foods[i])
        for i in range(0, self.person_max_count):
            self.moveObject(self.persons[i])
            self.moveResistanceProcess(self.persons[i])
            self.wallCollisitionProcess(self.persons[i])
            self.hungerProcess(self.persons[i])

        self.collisionProcess()

        for i in range(0, self.person_max_count):
            obj = self.persons[i]
            if controls[i].moveForward and not controls[i].moveBack:
                self.increaseSpeed(obj)
            if controls[i].moveBack and not controls[i].moveForward:
                self.decreaseSpeed(obj)
            if controls[i].rotateLeft and not controls[i].rotateRight:
                self.rotateLeft(obj)
            if controls[i].rotateRight and not controls[i].rotateLeft:
                self.rotateRight(obj)

    def collisionProcess(self):
        for i in range(0, self.food_max_count):
            for j in range(0, self.person_max_count):
                if self.isCollision(self.foods[i], self.persons[j]):
                    self.persons[j].hunger = self.persons[j].hunger - self.foods[i].calories * 0.005
                    if self.persons[j].hunger < 0:
                        self.persons[j].hunger = 0

                    # Перемещаем еду
                    point = self.getRandCoords()
                    self.foods[i] = Food(point[0], point[1], 10, 30)

    def isCollision(self, obj1: EnvObject, obj2: EnvObject) -> bool:
        return cycle_interception(obj1.x, obj1.y, obj1.width / 2, obj2.x, obj2.y, obj2.width / 2)

    def wallCollisitionProcess(self, obj: EnvObject):
        if self.isWallCollisition(obj):
            # Далеко не идеальное решение
            if obj.movementAngle >= 180:
                obj.movementAngle = obj.movementAngle - 180
                obj.movementSpeed = 0.1
            else:
                obj.movementAngle = obj.movementAngle + 180
                obj.movementSpeed = 0.1

    def isWallCollisition(self, obj: EnvObject) -> bool:
        radius = obj.width / 2
        if obj.x - radius <= 0:
            # TODO: костыль чтобы не проавливаться сквозь стены
            obj.x = 0 + radius
            return True
        if obj.x + radius >= self.max_x:
            # TODO: костыль чтобы не проавливаться сквозь стены
            obj.x = self.max_x - radius
            return True
        if obj.y - radius <= 0:
            # TODO: костыль чтобы не проавливаться сквозь стены
            obj.y = 0 + radius
            return True
        if obj.y + radius >= self.max_y:
            # TODO: костыль чтобы не проавливаться сквозь стены
            obj.y = self.max_y - radius
            return True
        return False

    # Здесь у меня возник небольшой внутренний спор,
    # выносить в EnvObject чтобы объект сам считал или все же среда считала?
    def moveObject(self, obj: EnvObject):
        angle = obj.movementAngle
        radius = obj.movementSpeed
        x = radius * sin(pi * 2 * angle / 360)
        y = radius * cos(pi * 2 * angle / 360)
        obj.x = x + obj.x
        obj.y = y + obj.y

    def increaseSpeed(self, obj: EnvObject):
        obj.movementSpeed = obj.movementSpeed + 0.01

    def decreaseSpeed(self, obj: EnvObject):
        obj.movementSpeed = obj.movementSpeed - 0.01

    def moveResistanceProcess(self, obj: EnvObject):
        obj.movementSpeed = obj.movementSpeed * 0.99

    def rotateRight(self, obj: EnvObject):
        obj.movementAngle = obj.movementAngle - 1.5
        if obj.movementAngle < 0:
            obj.movementAngle = 360 - obj.movementAngle

    def rotateLeft(self, obj: EnvObject):
        obj.movementAngle = obj.movementAngle + 1.5
        if obj.movementAngle > 360:
            obj.movementAngle = obj.movementAngle - 360

    def hungerProcess(self, obj: Person):
        obj.hunger = obj.hunger + 0.002
        if obj.hunger > 100:
            obj.hunger = 100

    def getRandCoords(self) -> [float, float]:
        return random() * (self.max_x - 60) + 30, random() * (self.max_y - 60) + 30
