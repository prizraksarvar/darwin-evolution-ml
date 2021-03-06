import torch
import torch.nn as nn
import torch.nn.functional as F


class SpinalCordNetwork(nn.Module):
    def __init__(self):
        super(SpinalCordNetwork, self).__init__()

        # На вход подаем angle, targetAngle, rotateDirection, speed, targetSpeed, distance, hunger
        self.layer1 = nn.Sequential(
            nn.Linear(3, 3),
            nn.Linear(3, 3),
            # nn.Dropout(),
            # nn.Tanh(),
            # nn.Linear(3, 3),
            nn.Linear(3, 2),
            # https://pytorch.org/docs/stable/generated/torch.nn.Softsign.html#torch.nn.Softsign
            # от -2 до + бесконечности
            nn.SELU(),
            # при -2 немного не достигает 0, при 0 будет 1, больше 0 примерно 1
            nn.Sigmoid()
        )
        # На выходе ожидаем forward, back, rotate_left, rotate_right

    # На вход подаем angle, targetAngle, rotateDirection, speed, targetSpeed, distance, hunger
    # На выходе ожидаем forward, back, rotate_left, rotate_right
    def forward(self, x) -> float:
        out = self.layer1(x)
        return out
