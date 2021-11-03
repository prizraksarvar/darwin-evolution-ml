class GameScore(object):
    die_count: int
    get_food_count: int
    loss: float

    def __init__(self):
        self.die_count = 0
        self.get_food_count = 0
        self.loss = 0
