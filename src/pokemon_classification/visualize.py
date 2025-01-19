import matplotlib.pyplot as plt
import torch
import typer
import numpy as np
from data import pokemon_data
from data import PokemonDataset
from model import MyAwesomeModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def evaluate():
    """Evaluate a trained model."""
    print("Evaluating pokemon model...")

    model = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(torch.load('models/model.pth', map_location=DEVICE))
    state_dict = torch.load('models/model.pth')
    print(type(state_dict))

    _, test_set = pokemon_data()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=1)

    model.eval()

    correct, total = 0, 0

    for img, target in test_dataloader:
        total += target.size(0)

    results = np.zeros((total, 1000))
    targets = np.zeros(total)
    i = 0

    for img, target in test_dataloader:
        img, target = img.to(DEVICE), target.to(DEVICE)
        y_pred = model(img)
        results[i,:] = y_pred.detach().numpy()
        targets[i] = target
        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        i += 1
    return total, correct, results, targets

def best_and_worst_pred(total, correct, results, targets) -> None:
    best_index = 0
    best = 0
    worst_index = 0
    worst = 1
    for i in range(total):
        if (np.argmax(results[i,:]) == targets[i] and np.argmax(results[i,:]) > best):
            best_index = i
            best = np.argmax(results[i,:])
        if (np.argmax(results[i,:]) != targets[i] and np.argmax(results[i,:]) < worst):
            worst_index = i
            worst = np.argmax(results[i,:])
    return

def main():
    typer.run(evaluate)

if __name__ == "__main__":
    main()
