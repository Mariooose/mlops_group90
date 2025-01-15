from pathlib import Path
import os
import pandas as pd
from sklearn.utils import shuffle
from torchvision.io import read_image

import typer
from torch.utils.data import Dataset
from torch import save
import torch

def create_dataframe(directory: Path) -> pd.DataFrame:
    "create a dataframe for data"
    data = []
    for subdir, dirs, files in os.walk(directory):
        label = os.path.basename(subdir)  # Assuming folder name is the label
        for file in files:
            img_path = os.path.join(subdir, file)
            data.append((img_path, label))
    
    df = pd.DataFrame(data, columns=['image_path', 'label'])
    df = shuffle(df).reset_index(drop=True)
    return df


class PokemonDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = annotations_file
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    test_frame = create_dataframe(f"{raw_data_path}/test")
    train_frame = create_dataframe(f"{raw_data_path}/train")
    validation_frame = create_dataframe(f"{raw_data_path}/val")
    test_data = PokemonDataset(test_frame,f"{raw_data_path}/test")
    train_data = PokemonDataset(train_frame,f"{raw_data_path}/train")
    validation_data = PokemonDataset(validation_frame,f"{raw_data_path}/val")
    save(test_data, f"{output_folder}/test_data.pt")
    save(train_data, f"{output_folder}/train_data.pt")
    save(validation_data, f"{output_folder}/validation_data.pt")

def pokemon_data():
    """Return train and test datasets for corrupt MNIST."""
    train_set = torch.load("data/processed/test_data.pt")
    test_set = torch.load("data/processed/train_data.pt")

    return train_set, test_set


if __name__ == "__main__":
    typer.run(preprocess)
    #pokemon_data()