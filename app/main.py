from contextlib import asynccontextmanager

import torch
import uvicorn
import typer
import pickle
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from pokemon_classification.model import MyAwesomeModel
from pokemon_classification.data import normalize


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up model on startup and shutdown."""
    global model, device
    print("Loading model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyAwesomeModel().to(device)
    model.load_state_dict(torch.load("models/model.pth", map_location=device, weights_only=True ))

    yield

    print("Cleaning up")
    del model, device


app = FastAPI(lifespan=lifespan)


@app.post("/classification/")
async def caption(data: UploadFile = File(...)):
    """Classifies an image of a pokemon."""
    i_image = Image.open(data.file)
    #if i_image.mode != "RGB":
    #    i_image = i_image.convert(mode="RGB")

    i_image = np.array(i_image)
    print(np.shape(i_image))
    i_image = torch.tensor(i_image)
    i_image = i_image.type(torch.FloatTensor)
    i_image = normalize(i_image)
    i_image = i_image.permute(2, 0, 1)

    img = torch.zeros(1,4,128,128,dtype=torch.float)
    img[0,:,:,:] = i_image

    model.eval()

    img = img.to(device)
    y_preds = model(img)
    pred = y_preds.argmax(dim=1).item()

    with open("pokemon_to_int.pkl", "rb") as f:
        pokemon_to_int = pickle.load(f)
    int_to_pokemon = {v: k for k, v in pokemon_to_int.items()}

    print(pred)

    return int_to_pokemon[pred]


if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)