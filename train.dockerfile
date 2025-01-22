# Base image
FROM python:3.11-slim

# Install Python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*


# copy essential parts of the computer to container
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src src/
COPY data/ data/
COPY pokemon_to_int.pkl pokemon_to_int.pkl
COPY data/processed/ data/processed/
COPY models/model.pth models/model.pth

#working directory

WORKDIR /
# RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "src/pokemon_classification/train.py"]
