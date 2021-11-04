class BaseLearner:
    def gameRestarted(self):
        pass

    def learnLoop(self):
        pass

    def testLoop(self):
        pass

    def done(self):
        pass

    # Дисконтированная награда
    def get_corrected_y(self, v: float, predY: float, gamma=0.98) -> float:
        running_add = self.calcRewardFunc(v)
        running_add_negative = 2 - running_add

        max_val = 0.60
        min_val = 0.40

        vt = predY
        it = predY
        if it > 0.5:
            vt = it * running_add
        else:
            vt = (it if it >= min_val else min_val) * running_add_negative

        if vt > max_val:
            vt = max_val

        if vt < min_val:
            vt = min_val

        return vt

    def calcRewardFunc(self, v: float) -> float:
        return 1.0 + v
