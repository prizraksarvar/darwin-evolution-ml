import torch.nn as nn
import torch.nn.functional as F


class SpinalCordNetwork(nn.Module):
    def __init__(self):
        super(SpinalCordNetwork, self).__init__()

        # На вход подаем angle, targetAngle, rotateDirection, speed, targetSpeed, distance
        self.layer1 = nn.Linear(6, 8)
        self.layer2 = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
        )
        self.layer3 = nn.Linear(16, 8)
        self.layer4 = nn.Sequential(
            nn.Linear(8, 4),
        )
        # https://pytorch.org/docs/stable/generated/torch.nn.Softsign.html#torch.nn.Softsign
        self.layer5 = nn.Softplus()
        # На выходе ожидаем forward, back, rotate_left, rotate_right

    # На вход подаем angle, targetAngle, rotateDirection, speed, targetSpeed, distance
    # На выходе ожидаем forward, back, rotate_left, rotate_right
    def forward(self, x) -> float:
        out = self.layer1(x)
        out = F.relu(self.layer2(out))
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out
