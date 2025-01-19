import torch
import typer
from pokemon_classification.model import MyAwesomeModel
import os

from pokemon_classification.data import PokemonDataset, pokemon_data
# Detect if running in Docker
RUNNING_IN_DOCKER = os.path.exists("/.dockerenv")
DEVICE = torch.device("cpu")
#DEVICE = torch.device("cpu" if RUNNING_IN_DOCKER else 
#                      "cuda" if torch.cuda.is_available() else 
 #                     "mps" if torch.backends.mps.is_available() else 
 #                     "cpu")
##DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def evaluate() -> None:
    """Evaluate a trained model."""
    print("Evaluating pokemon model...")

    model = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(torch.load("models/model.pth", map_location=DEVICE, weights_only=True ))
    state_dict = torch.load("models/model.pth",weights_only=True)
    print(type(state_dict))

    _, test_set = pokemon_data()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    model.eval()
    correct, total = 0, 0
    for img, target in test_dataloader:
        img, target = img.to(DEVICE), target.to(DEVICE)
        y_pred = model(img)
        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)
    print(f"Test accuracy: {correct / total}")


def main():
    typer.run(evaluate)


if __name__ == "__main__":
    main()
