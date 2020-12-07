import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LinearMutation(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)
