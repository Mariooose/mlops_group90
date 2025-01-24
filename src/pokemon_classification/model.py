import timm
import torch
from torch import nn

from src.my_logger import logger

# create model with output size 1000 and takes in 4 channels
resnet18 = timm.create_model("resnet18", num_classes=1000, in_chans=4, pretrained=True)

if __name__ == "__main__":
    # model = MyAwesomeModel()
    model = resnet18
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 4, 128, 128)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
