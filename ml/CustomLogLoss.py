import torch
from torch import Tensor, nn
from torch.nn import Module


class CustomLogLoss(nn.Module):
    def __init__(self) -> None:
        super(CustomLogLoss, self).__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # Этап 1 - логарифмируем вероятности действий
        # prob = torch.log(pred[np.arange(len(y)), y])
        prob = torch.log(input)
        # Этап 2 - отрицательное среднее произведения вероятностей на награду
        selected_probs = target * prob
        loss = -selected_probs.mean()
        return loss
