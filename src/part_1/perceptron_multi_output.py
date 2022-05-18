import torch
from torch import Tensor


class Perceptron:
    def __init__(self, num_inputs: int, num_outputs: int):
        self.weights = torch.randn(num_inputs, num_outputs)
        self.bias = torch.randn(num_outputs)

    def forward(self, x: Tensor) -> Tensor:
        # Examining how matrix multiplication works here:
        #
        #   tensor             | shape
        #   -------------------|--------
        #   x                  | (num_inputs,)
        #   weights            | (num_inputs, num_outputs)
        #   bias               | (num_outputs,)
        #                      |
        #   x @ weights        | (num_outputs,)   --> matrix multiply
        #   x @ weights + bias | (num_outputs,)   --> matrix multiply, then add

        return x @ self.weights + self.bias


if __name__ == "__main__":
    perceptron = Perceptron(num_inputs=3, num_outputs=2)
    inputs = torch.randn(3)
    output = perceptron(inputs)
    # 'output' should now contain two values.  Example:
    #   tensor([0.2185, 0.7698])
