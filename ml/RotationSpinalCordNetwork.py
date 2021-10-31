import torch.nn as nn
import torch.nn.functional as F


class RotationSpinalCordNetwork(nn.Module):
    def __init__(self):
        super(RotationSpinalCordNetwork, self).__init__()

        # На вход подаем angleDiff, toLeft(0/1), toRight(0/1)
        self.layer1 = nn.Linear(3, 6)
        self.layer2 = nn.Dropout()
        self.layer3 = nn.Linear(6, 2)
        self.layer4 = nn.Sigmoid()
        # На выходе ожидаем rotateLeft, rotateRight

    # На вход подаем angle, targetAngle, rotateDirection, speed, targetSpeed, distance, hunger
    # На выходе ожидаем forward, back, rotate_left, rotate_right
    def forward(self, x) -> float:
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out
