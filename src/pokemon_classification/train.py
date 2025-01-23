import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, accuracy_score, f1_score, precision_score, recall_score
import torch
import typer
from pokemon_classification.model import MyAwesomeModel
from pokemon_classification.model import resnet18
from my_logger import logger
from pokemon_classification.data import pokemon_data
import sys
import os
import wandb


from dotenv import load_dotenv

load_dotenv()
import os

api_key = os.getenv("WANDB_API_KEY")
wandb_project = os.getenv("WANDB_PROJECT")
wandb_entity = os.getenv("WANDB_ENTITY")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
# DEVICE = torch.device("cpu")


def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 10, run_wandb: int = 0) -> None:
    """Train a model on pokemon."""
    print("Training started")
    logger.info("Training started")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    if run_wandb:
        run = wandb.init(
            project=wandb_project, config={"lr": lr, "batch_size": batch_size, "epochs": epochs}, entity=wandb_entity
        )

    logger.debug(f"{lr=}, {batch_size=}, {epochs=}")

    logger.info("Fetching model and training data")
    model = resnet18.to(DEVICE)
    train_set, _ = pokemon_data()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(epochs):
        model.train()

        preds, targets = [], []
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            if run_wandb:
                wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})
            statistics["train_accuracy"].append(accuracy)

            preds.append(y_pred.detach().cpu())
            targets.append(target.detach().cpu())

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

                if run_wandb:
                    # add a plot of the input images
                    images = wandb.Image(img[:5].detach().cpu(), caption="Input images")
                    wandb.log({"images": images})

                    # add a plot of histogram of the gradients
                    grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None], 0)
                    wandb.log({"gradients": wandb.Histogram(grads.cpu())})

            logger.success(f"Epoch {epoch} done")

        if run_wandb:
            # add a custom matplotlib plot of the ROC curves
            preds = torch.cat(preds, 0)
            targets = torch.cat(targets, 0)

            for class_id in range(1000):
                one_hot = torch.zeros_like(targets)
                one_hot[targets == class_id] = 1
                _ = RocCurveDisplay.from_predictions(
                    one_hot,
                    preds[:, class_id],
                    name=f"ROC curve for {class_id}",
                    plot_chance_level=(class_id == 2),
                )

            wandb.log({"roc": wandb.Image(plt)})
            # wandb.plot({"roc": plt})
            plt.close()  # close the plot to avoid memory leaks and overlapping figures

    if run_wandb:
        final_accuracy = accuracy_score(targets, preds.argmax(dim=1))
        final_precision = precision_score(targets, preds.argmax(dim=1), average="weighted")
        final_recall = recall_score(targets, preds.argmax(dim=1), average="weighted")
        final_f1 = f1_score(targets, preds.argmax(dim=1), average="weighted")

        # first we save the model to a file then log it as an artifact
        torch.save(model.state_dict(), "model.pth")
        artifact = wandb.Artifact(
            name="corrupt_mnist_model",
            type="model",
            description="A model trained to classify corrupt MNIST images",
            metadata={"accuracy": final_accuracy, "precision": final_precision, "recall": final_recall, "f1": final_f1},
        )
        artifact.add_file("model.pth")
        run.log_artifact(artifact)

    print("Training complete")
    logger.success("Training complete")

    try:
        torch.save(model.state_dict(), "models/model.pth")
    except:
        logger.error("Failed to save model")
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
