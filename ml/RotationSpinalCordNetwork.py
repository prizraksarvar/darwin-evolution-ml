import torch.nn as nn
import torch.nn.functional as F


class RotationSpinalCordNetwork(nn.Module):
    def __init__(self):
        super(RotationSpinalCordNetwork, self).__init__()

        # На вход подаем angleDiff, toLeft(0/1), toRight(0/1)
        self.layer1 = nn.Sequential(
            nn.Linear(2, 2, bias=True),
            nn.Linear(2, 2, bias=True),
            nn.Dropout(),
            nn.Tanh(),
            # nn.Linear(2, 2, bias=True),
            nn.Linear(2, 2),
            # https://pytorch.org/docs/stable/generated/torch.nn.Softsign.html#torch.nn.Softsign
            # от -2 до + бесконечности
            nn.SELU(),
            # при -2 немного не достигает 0, при 0 будет 1, больше 0 примерно 1
            nn.Sigmoid()
        )
        # На выходе ожидаем rotateLeft, rotateRight

    # На вход подаем angle, targetAngle, rotateDirection, speed, targetSpeed, distance, hunger
    # На выходе ожидаем forward, back, rotate_left, rotate_right
    def forward(self, x) -> float:
        out = self.layer1(x)
        return out
