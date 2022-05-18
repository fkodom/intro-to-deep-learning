import torch
from torch import Tensor, nn


class MultiLayerPerceptron(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        num_features: int = 64,
    ):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, num_features)
        self.linear2 = nn.Linear(num_features, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        y1 = self.linear1(x)
        act = self.relu(y1)
        y2 = self.linear2(act)
        return y2


if __name__ == "__main__":
    model = MultiLayerPerceptron(
        num_inputs=10,
        num_outputs=2,
        num_features=32,  # optional
    )
    inputs = torch.randn(16, 10)
    output = model(inputs)
    # 'output' has shape: (16, 2)
