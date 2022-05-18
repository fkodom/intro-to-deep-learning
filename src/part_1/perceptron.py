import torch
from torch import Tensor


class Perceptron:
    def __init__(self, num_inputs: int):
        # In our initial examples, we randomly choose the weights. You could
        # manually define them for specific problems, such as the "festival"
        # example above. Later on, we'll learn how to "train" the weights for
        # more generic applications.
        self.weights = torch.randn(num_inputs)
        self.bias = torch.randn(1)

    def __call__(self, x: Tensor) -> Tensor:
        # Perform dot product with 'self.weights' then add 'self.bias'
        return x @ self.weights + self.bias


if __name__ == "__main__":
    # --- Example usage with 3 inputs ---
    perceptron = Perceptron(num_inputs=3)
    inputs = torch.randn(3)
    output = perceptron(inputs)
    # 'output' just contains one number.  Example:
    #   tensor([-0.1493])
