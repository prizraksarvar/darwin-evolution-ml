import torch.nn as nn
import torch.nn.functional as F


class SpinalCordNetwork(nn.Module):
    def __init__(self):
        super(SpinalCordNetwork, self).__init__()

        # На вход подаем angle, targetAngle, rotateDirection, speed, targetSpeed, distance, hunger
        self.layer1 = nn.Linear(5, 16)
        self.layer2 = nn.Linear(16, 24)
        self.layer3 = nn.Sequential(
            nn.Linear(24, 16),
            nn.ReLU(),
        )
        self.layer4 = nn.Linear(16, 8)
        self.layer5 = nn.Sequential(
            nn.Linear(8, 2),
        )
        # https://pytorch.org/docs/stable/generated/torch.nn.Softsign.html#torch.nn.Softsign
        self.layer6 = nn.Softplus()
        # На выходе ожидаем forward, back, rotate_left, rotate_right

    # На вход подаем angle, targetAngle, rotateDirection, speed, targetSpeed, distance, hunger
    # На выходе ожидаем forward/back, rotate_left/rotate_right
    def forward(self, x) -> float:
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        return out
