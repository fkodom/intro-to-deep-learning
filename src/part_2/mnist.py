from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


def get_mnist_dataset(train: bool = True) -> MNIST:
    return MNIST(
        root="./data/mnist",
        train=train,
        download=True,
        transform=ToTensor(),
    )


def get_mnist_dataloader(train: bool = True, batch_size: int = 32) -> DataLoader:
    return DataLoader(
        get_mnist_dataset(train=train),
        batch_size=batch_size,
        shuffle=train,
        drop_last=train,
    )


class MultiLayerPerceptron(nn.Module):
    """Copy-pasta from last time 🎉"""

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        num_features: int = 256,
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


class MnistMLP(nn.Module):
    def __init__(self, num_features: int = 256):
        super().__init__()
        self.mlp = MultiLayerPerceptron(
            # Each image has (28 * 28) = 784 pixels
            num_inputs=784,
            # There are 10 different output classes (digits 0-9)
            num_outputs=10,
            num_features=num_features,
        )

    def forward(self, x: Tensor) -> Tensor:
        # Important change -- each digit is a 28x28 image, so we flatten them
        # into vectors of length (28 * 28) = 784. We could handle this
        # externally, but it's nice that the model does it automatically.
        #
        # Starts at 'start_dim=1', so that we flatten each individual
        # sample in the mini-batch.  Starting at 'start_dim=0' would flatten
        # the entire batch into a single vector, which is not what we want.
        x = x.flatten(start_dim=1)
        y = self.mlp(x)
        return y


def one_hot(labels: Tensor, num_classes: int) -> Tensor:
    class_labels = torch.arange(num_classes)
    one_hot_labels = labels.reshape(-1, 1) == class_labels.reshape(1, -1)
    return one_hot_labels


def loss_function(scores: Tensor, labels: Tensor) -> Tensor:
    one_hot_labels = one_hot(labels, num_classes=10)
    # MSE is already implemented in PyTorch, along with a bunch of other
    # common loss functions.  Let's use the official implementation.
    #
    # Also note -- 'mse_loss' expects both input Tensors to have 'float' types.
    # 'one_hot' produces a boolean array, so convert it to float first.
    loss = F.mse_loss(scores, one_hot_labels.float())
    return loss


def zero_grads(model: nn.Module):
    for parameter in model.parameters():
        parameter.grad = None


def zero_gradients(model: nn.Module):
    """Zero out all existing gradients, so that PyTorch doesn't just add to them
    each time we call 'loss.backward(). A technicality of working with PyTorch.
    """
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.zero_()


def sgd_update(
    model: nn.Module,
    batch: Tuple[Tensor, Tensor],
    learning_rate: float = 1e-2,
) -> float:
    # Technicality -- we have to "zero out" all of the existing gradients.
    # Otherwise, PyTorch will carry them over from the last update, and every call
    # to 'loss.backward()' will just add to the existing gradients.
    zero_gradients(model)

    inputs, labels = batch
    scores = model(inputs)
    loss = loss_function(scores, labels)
    loss.backward()

    for parameter in model.parameters():
        # For in-place updates, adjust the 'parameter.data' property.
        parameter.data -= parameter.grad * learning_rate

    return float(loss)


def test_accuracy(model: nn.Module, test_dataloader: DataLoader) -> float:
    """Computes the average accuracy across all batches in the DataLoader"""

    accuracies = []
    for batch in test_dataloader:
        inputs, labels = batch
        scores = model(inputs)
        accuracy = (scores.argmax(dim=-1) == labels).float().mean()
        accuracies.append(float(accuracy))

    return sum(accuracies) / len(accuracies)


model = MnistMLP()
train_dataloader = get_mnist_dataloader(train=True, batch_size=64)

# Train for 20 epochs
for epoch in range(20):
    # Aggregate losses from all mini-batches, and only print the
    # average loss at the end of each epoch.
    losses = []

    for batch in train_dataloader:
        loss = sgd_update(model, batch, learning_rate=1e-2)
        losses.append(loss)

    avg_loss = sum(losses) / len(losses)
    print(f"Epoch {epoch}: Loss = {avg_loss:.3f}")

# It's standard to test the model with 'batch_size' of 1.
test_dataloader = get_mnist_dataloader(train=False, batch_size=1)
accuracy = test_accuracy(model, test_dataloader)
print(f"Test accuracy: {accuracy:.3f}")

# Results from my local run:
#   Epoch 0: Loss = 0.072
#   Epoch 1: Loss = 0.052
#   Epoch 2: Loss = 0.046
#   Epoch 3: Loss = 0.042
#   Epoch 4: Loss = 0.040
#   Epoch 5: Loss = 0.038
#   Epoch 6: Loss = 0.036
#   Epoch 7: Loss = 0.034
#   Epoch 8: Loss = 0.033
#   Epoch 9: Loss = 0.031
#   Epoch 10: Loss = 0.030
#   Epoch 11: Loss = 0.029
#   Epoch 12: Loss = 0.028
#   Epoch 13: Loss = 0.027
#   Epoch 14: Loss = 0.026
#   Epoch 15: Loss = 0.026
#   Epoch 16: Loss = 0.025
#   Epoch 17: Loss = 0.024
#   Epoch 18: Loss = 0.024
#   Epoch 19: Loss = 0.023
#   Test accuracy: 0.924
