import torch.nn as nn
import torch.nn.functional as F


class SpinalCordNetwork(nn.Module):
    def __init__(self):
        super(SpinalCordNetwork, self).__init__()

        # На вход подаем angle, targetAngle, rotateDirection, speed, targetSpeed, distance, hunger
        self.layer1 = nn.Linear(5, 15)
        self.layer2 = nn.Sequential(
            nn.Linear(15, 15),
            # nn.ReLU(),
        )
        self.layer3 = nn.Dropout()
        self.layer4 = nn.Linear(15, 15)
        self.layer5 = nn.Linear(15, 4)
        # https://pytorch.org/docs/stable/generated/torch.nn.Softsign.html#torch.nn.Softsign
        # от -2 до + бесконечности
        self.layer6 = nn.SELU()
        # при -2 немного не достигает 0, при 0 будет 1, больше 0 примерно 1
        self.layer7 = nn.Sigmoid()
        # На выходе ожидаем forward, back, rotate_left, rotate_right

    # На вход подаем angle, targetAngle, rotateDirection, speed, targetSpeed, distance, hunger
    # На выходе ожидаем forward, back, rotate_left, rotate_right
    def forward(self, x) -> float:
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        return out
