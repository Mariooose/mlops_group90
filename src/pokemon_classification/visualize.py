import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import typer

from data import create_dataframe, pokemon_data
from my_logger import logger
from pokemon_classification.model import resnet18

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def visualize():
    """Visualize results of midel."""
    print("Visualizing  pokemon model...")
    logger.info("Visualizing pokemon model - fetching model")
    model = resnet18.to(DEVICE)
    model.load_state_dict(torch.load("models/model.pth", map_location=DEVICE, weights_only=True))

    logger.info("Loading test data")
    _, test_set = pokemon_data()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=1)

    model.eval()

    with open("pokemon_to_int.pkl", "rb") as f:
        pokemon_to_int = pickle.load(f)
    int_to_pokemon = {v: k for k, v in pokemon_to_int.items()}

    correct, total = 0, 0

    for img, target in test_dataloader:
        total += target.size(0)

    print(total)

    results = np.zeros((total, 1000))
    targets = np.zeros(total)
    images = np.zeros((128, 128, 4, total))
    preds = np.zeros(total)

    i = 0

    for img, target in test_dataloader:
        image = img.permute(2, 3, 1, 0).cpu().numpy().squeeze()
        images[:, :, :, i] = image
        targets[i] = target.item()

        img, target = img.to(DEVICE), target.to(DEVICE)
        with torch.no_grad():
            y_pred = model(img)
        y_pred = torch.nn.functional.softmax(y_pred[0], dim=0)
        results[i, :] = y_pred.detach().cpu().numpy()
        preds[i] = y_pred.argmax()
        correct += (y_pred.argmax() == target).float().sum().item()
        i += 1

    logger.info("Visualizing analysis done!")
    logger.info("Creating best and worst prediction...")
    best_and_worst_pred(images, total, correct, results, preds, targets, int_to_pokemon)
    logger.success("Best and worst prediction done")
    logger.info("Creating best and worst class...")
    best_and_worst_class(images, total, correct, results, preds, targets, int_to_pokemon)
    logger.success("Best and worst class done!")
    return


def best_and_worst_pred(images, total, correct, results, preds, targets, int_to_pokemon) -> None:
    best_index = 0
    best = 0
    worst_index = 0
    worst = 0
    for i in range(total):
        if np.argmax(results[i, :]) == targets[i] and np.max(results[i, :]) > best:
            best_index = i
            best = np.max(results[i, :])
        if np.argmax(results[i, :]) != targets[i] and np.max(results[i, :]) > worst:
            worst_index = i
            worst = np.max(results[i, :])

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(images[:, :, :, best_index].squeeze())
    plt.title(
        f"Prediction: {int_to_pokemon[preds[best_index]]}\nTrue: {int_to_pokemon[targets[best_index]]}\nConfidence: {best}"
    )
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(images[:, :, :, worst_index].squeeze())
    plt.title(
        f"Prediction: {int_to_pokemon[preds[worst_index]]}\nTrue: {int_to_pokemon[targets[worst_index]]}\nConfidence: {worst}"
    )
    plt.axis("off")
    plt.show()

    return


def best_and_worst_class(images, total, correct, results, preds, targets, int_to_pokemon) -> None:
    total_per_pokemon = np.zeros(1000)
    correct_per_pokemon = np.zeros(1000)

    for i in range(total):
        total_per_pokemon[int(targets[i])] += 1
        if preds[i] == targets[i]:
            correct_per_pokemon[int(targets[i])] += 1

    class_error_rate = correct_per_pokemon / total_per_pokemon
    argsort_class_error_rate = np.argsort(class_error_rate)
    best = np.zeros(10)
    best_labels = []
    worst = np.zeros(10)
    worst_labels = []
    for i in range(10):
        best[i] = class_error_rate[argsort_class_error_rate[-i]]
        best_labels.append(int_to_pokemon[argsort_class_error_rate[-i]])
        worst[i] = class_error_rate[argsort_class_error_rate[i]]
        worst_labels.append(int_to_pokemon[argsort_class_error_rate[i]])

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.bar(best_labels, best)
    plt.title("Top 10 best classes")
    plt.subplot(1, 2, 2)
    plt.bar(worst_labels, worst)
    plt.title("Top 10 worst classes")
    plt.show()

    return


def main():
    typer.run(visualize)


if __name__ == "__main__":
    main()
