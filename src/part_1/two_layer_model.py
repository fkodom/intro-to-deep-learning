import torch
from torch import Tensor, nn


class TwoLayerModel(nn.Module):
    # 'Module' is the parent class of 'nn.Linear' and essentially every
    # other type of neural network layer in PyTorch. We only need minor
    # changes from before, and it will make our lives easier later on.

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        num_features: int = 64,
    ):
        # Must run the '__init__' method before defining layers.
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, num_features)
        self.linear2 = nn.Linear(num_features, num_outputs)

    def forward(self, x: Tensor) -> Tensor:
        # For 'Module' objects, the 'forward' method defines how
        # data moves through the layers.

        # Input shape: (batch_size, num_inputs)
        y1 = self.linear1(x)  # shape: (batch_size, num_features)
        y2 = self.linear2(y1)  # shape: (batch_size, num_outputs)
        return y2


if __name__ == "__main__":
    model = TwoLayerModel(
        num_inputs=10,
        num_outputs=2,
        num_features=32,  # optional
    )
    inputs = torch.randn(16, 10)
    output = model(inputs)
    # 'output' has shape: (16, 2)
