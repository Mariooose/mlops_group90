import torch
from torch import nn
from my_logger import logger


class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, 3, 1)  # Accepts 3 input channels
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(25088, 1000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #raise error
        if x.ndim != 4:
            logger.error('Expected input to a 4D tensor')
            raise ValueError('Expected input to a 4D tensor')
        if x.shape[1] != 4 or x.shape[2] != 128 or x.shape[3] != 128:
            logger.error('Expected sample to have shape 4,128,128')
            raise ValueError('Expected sample to have shape 4,128,128')

        """Forward pass."""
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        result = self.fc1(x)
        return result


if __name__ == "__main__":
    model = MyAwesomeModel()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 4, 128, 128)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
