import matplotlib.pyplot as plt
import torch
import typer
from pokemon_classification.model import MyAwesomeModel
from my_logger import logger
from pokemon_classification.data import pokemon_data
import sys
import os


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
#DEVICE = torch.device("cpu")

def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 10) -> None:
    """Train a model on pokemon."""
    print("Training started")
    logger.info("Training started")
    print(f"{lr=}, {batch_size=}, {epochs=}")
    logger.debug(f"{lr=}, {batch_size=}, {epochs=}")

    logger.info('Fetching model and training data')
    model = MyAwesomeModel().to(DEVICE)
    train_set, _ = pokemon_data()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(epochs):
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

        logger.success(f'Epoch {epoch} done')

    print("Training complete")
    logger.success("Training complete")

    try:
        torch.save(model.state_dict(), "models/model.pth")
    except:
        logger.error('Failed to save model')
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("reports/figures/training_statistics.png")


def main():
    typer.run(train)

if __name__ == "__main__":
    main()
