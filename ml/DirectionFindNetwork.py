import torch.nn as nn


class DirectionFindNetwork(nn.Module):
    def __init__(self):
        super(DirectionFindNetwork, self).__init__()
        self.layer1 = nn.Linear(4, 8)
        self.layer2 = nn.Linear(8, 8)
        self.layer3 = nn.Linear(8, 8)
        self.layer4 = nn.Linear(8, 1)

    def forward(self, x) -> float:
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out
