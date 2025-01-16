import os
import pickle
from pathlib import Path
#test
import pandas as pd
import torch
import typer
from PIL import Image
from sklearn.utils import shuffle
from torch import save
from torch.utils.data import Dataset
from torchvision.io import decode_image

with open("pokemon_to_int.pkl", "rb") as f:
    pokemon_to_int = pickle.load(f)


def create_dataframe(directory: Path) -> pd.DataFrame:
    "create a dataframe for data"
    data = []
    for subdir, dirs, files in os.walk(directory):
        label = os.path.basename(subdir)  # Assuming folder name is the label
        for file in files:
            if file != ".DS_Store":
                img_path = os.path.join(subdir, file)
                data.append((img_path, label))


    df = pd.DataFrame(data, columns=["image_path", "label"])
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
        # print(img_path)
        image = decode_image(img_path, mode="RGB").type(torch.float32)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        label = pokemon_to_int[label]
        return image, label


def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    test_frame = create_dataframe(f"{raw_data_path}/test")
    train_frame = create_dataframe(f"{raw_data_path}/train")
    validation_frame = create_dataframe(f"{raw_data_path}/val")

    test_frame.to_csv(f"{output_folder}/test_frame.csv",index=False)
    train_frame.to_csv(f"{output_folder}/train_frame.csv",index=False)
    validation_frame.to_csv(f"{output_folder}/validation_frame.csv",index=False)


def pokemon_data():
    """Return train and test datasets for pokemon classification."""
    train_frame = pd.read_csv("data/processed/train_frame.csv")
    test_frame = pd.read_csv("data/processed/test_frame.csv")

    train_set = PokemonDataset(train_frame,"")
    test_set = PokemonDataset(test_frame,"")

    return train_set, test_set


if __name__ == "__main__":
    typer.run(preprocess)
