import pytest
from pokemon_classification.model import MyAwesomeModel
import torch

def test_model():
    model = MyAwesomeModel()
    x = torch.randn(1, 4, 128, 128)
    y = model(x)
    assert y.shape == (1, 1000)

def test_error_on_wrong_shape():
    model = MyAwesomeModel()
    with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
        model(torch.randn(1,2,3))
    with pytest.raises(ValueError, match='Expected sample to have shape 4,128,128'):
        model(torch.randn(1,4,128,129))
