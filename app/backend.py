from contextlib import asynccontextmanager

import anyio
import torch
import uvicorn
import pickle
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from src.pokemon_classification.model import resnet18
from torchvision import transforms

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up model on startup and shutdown."""
    global model, DEVICE, transform, int_to_pokemon
    print("Loading model")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18.to(DEVICE)
    model.load_state_dict(torch.load("models/model.pth", map_location=DEVICE, weights_only=True ))

    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ],
    )

    with open("pokemon_to_int.pkl", "rb") as f:
        pokemon_to_int = pickle.load(f)
    int_to_pokemon = {v: k for k, v in pokemon_to_int.items()}

    yield

    print("Cleaning up")
    del model, DEVICE, transform, int_to_pokemon


app = FastAPI(lifespan=lifespan)

def predict_pokemon(image_path: str) -> str:
    """Predict pokemon from image"""

    i_image = Image.open(image_path)

    i_image = transform(i_image)
    i_image = (i_image - i_image.mean()) / i_image.std()

    img = torch.zeros(1,4,128,128,dtype=torch.float)
    img[0,:,:,:] = i_image

    model.eval()

    img = img.to(DEVICE)
    y_preds = model(img)
    pred = y_preds.argmax(dim=1).item()

    return int_to_pokemon[pred]

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Hello from the backend!"}

@app.post("/classify/")
async def classify_pokemon(data: UploadFile = File(...)):
    """Classify image endpoint"""

    try:
        contents = await data.read(-1)

        async with await anyio.open_file(data.filename, "wb") as f:
            await f.write(contents)

        pred = predict_pokemon(data.filename)

        return {"filename": data.filename, "prediction": pred}
    except Exception as e:
        raise HTTPException(status_code=500) from e


if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)