import random
import numpy as np

from locust import HttpUser, between, task


class MyUser(HttpUser):
    """A simple Locust user class that defines the tasks to be performed by the users."""

    wait_time = between(1, 2)

    @task
    def get_root(self) -> None:
        """A task that simulates a user visiting the root URL of the FastAPI app."""
        self.client.get("/")

    @task(3)
    def post_image(self) -> None:
        """A task that simulates a user uploading a random PNG to the FastAPI app."""
        random_png = np.random.rand(128, 128, 4)
        self.client.post("/classify/", files={"data": random_png.tobytes()}, timeout=10)
