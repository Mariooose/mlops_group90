from fastapi.testclient import TestClient
from src.pokemon_classification.api import app

client = TestClient(app)

def test_read_root(model):
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Welcome to the MNIST model inference API!"}