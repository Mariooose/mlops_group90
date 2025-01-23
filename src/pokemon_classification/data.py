import os
import pickle
from pathlib import Path

# test
import pandas as pd
import torch
import typer
from PIL import Image
from sklearn.utils import shuffle
from torch import save
from torch.utils.data import Dataset
from torchvision.io import decode_image
import dvc.api
import sys
import os
from my_logger import logger  # Import the logger


# Read pokemon_to_int dictiionary that converts a label to a number
logger.info("Reading pokemon_to_int.pkl")
with open("pokemon_to_int.pkl", "rb") as f:
    pokemon_to_int = pickle.load(f)


def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize images."""
    return (images - images.mean()) / images.std()


# creates a pd dataframe that has an imagepath associated with a label
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
        image = decode_image(img_path, mode="RGB").type(torch.float32)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        label = pokemon_to_int[label]
        return image, label


def preprocess(raw_data_path: Path, output_folder: Path, download_data=False) -> None:
    print("Preprocessing data...")
    logger.info("Preprocessing data")
    test_frame = create_dataframe(f"{raw_data_path}/test")
    train_frame = create_dataframe(f"{raw_data_path}/train")
    validation_frame = create_dataframe(f"{raw_data_path}/val")
    frames = [test_frame, train_frame, validation_frame]

    for j, data_type in enumerate(["test", "train", "validation"]):
        # load data into one tensor and labels into one array
        current_frame = frames[j]
        n = len(current_frame)
        images = torch.zeros(n, 4, 128, 128, dtype=torch.float)
        labels = []
        for i in range(n):
            path, label = current_frame.iloc[i, :]  # get image path and label from dataframe
            img = decode_image(path).type(torch.float)  # read the image and change datatype to float
            images[i, :, :, :] = normalize(img)  # normalize image and save it to the big tensor
            labels.append(pokemon_to_int[label])  # convert label into number and append it to label

        # Check if the folder exists if not creates it
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # save data and labels
        torch.save(images, f"{output_folder}/{data_type}_images.pt")
        torch.save(torch.tensor(labels), f"{output_folder}/{data_type}_target.pt")  # convert labels into tensor

        print(f"processed {data_type}")
        logger.success(f"Processed {data_type}")


def pokemon_data():
    """Return train and test datasets for pokemon classification."""
    logger.info("Loading data via pokemon_data()...")

    train_images, train_target, test_images, test_target = None, None, None, None
    # check if data is locally else fetch from GCP bucket
    try:
        list_of_files = os.listdir("data/processed")
    except FileNotFoundError:
        list_of_files = []

    if set(["train_images.pt", "train_target.pt", "test_images.pt", "test_target.pt"]).issubset(list_of_files):
        train_images = torch.load("data/processed/train_images.pt", weights_only=True)
        train_target = torch.load("data/processed/train_target.pt", weights_only=True)
        test_images = torch.load("data/processed/test_images.pt", weights_only=True)
        test_target = torch.load("data/processed/test_target.pt", weights_only=True)
    else:
        train_images = torch.load("/gcs/mlops_project90/data/processed/train_images.pt", weights_only=True)
        train_target = torch.load("/gcs/mlops_project90/data/processed/train_target.pt", weights_only=True)
        test_images = torch.load("/gcs/mlops_project90/data/processed/test_images.pt", weights_only=True)
        test_target = torch.load("/gcs/mlops_project90/data/processed/test_target.pt", weights_only=True)

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    logger.success("Data loaded successfully")
    return train_set, test_set


if __name__ == "__main__":
    typer.run(preprocess)
