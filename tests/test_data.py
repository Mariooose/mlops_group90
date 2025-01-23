import pytest
from torch.utils.data import Dataset
from pokemon_classification.data import pokemon_data


def test_my_dataset():
    print("test")

    """Test the MyDataset class."""
    trainset, testset = pokemon_data()
    print("im here")

    # test if train and test sets are of type Dataset
    assert isinstance(trainset, Dataset)
    assert isinstance(testset, Dataset)

    # test if train and test sets are correct length


if __name__ == "__main__":
    test_my_dataset()
