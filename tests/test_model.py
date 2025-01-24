import pytest
import torch

from pokemon_classification.model import resnet18


def test_model():
    model = resnet18
    x = torch.randn(1, 4, 128, 128)
    y = model(x)
    assert y.shape == (1, 1000)



def test_error_on_wrong_shape():
    model = resnet18
    with pytest.raises(RuntimeError):
        model(torch.randn(1, 2, 3))
