FROM python:3.11-slim

EXPOSE $PORT

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir /app

WORKDIR /app

COPY app/requirements_backend.txt app/requirements_backend.txt
COPY pokemon_to_int.pkl pokemon_to_int.pkl
COPY models/model.pth models/model.pth
COPY app/backend.py app/backend.py
COPY src/pokemon_classification/model.py src/pokemon_classification/model.py
COPY src/my_logger.py src/my_logger.py

RUN --mount=type=cache,target=/root/.cache/pip pip install -r app/requirements_backend.txt

CMD exec uvicorn --port $PORT --host 0.0.0.0 app.backend:app