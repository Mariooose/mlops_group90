import torch
import typer
from data import pokemon_data
from model import MyAwesomeModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def evaluate() -> None:
    """Evaluate a trained model."""
    print("Evaluating pokemon model...")

    model = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(torch.load('models/model.pth', map_location=DEVICE))
    state_dict = torch.load('models/model.pth')
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