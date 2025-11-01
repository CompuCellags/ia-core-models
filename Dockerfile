FROM python:3.14-slim

LABEL maintainer="Develop Aguascalientes"
LABEL version="1.0"
LABEL description="Entorno reproducible para entrenamiento de modelos IA educativos e industriales"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY configs/requirements.txt ./requirements.txt

RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && rm -rf ~/.cache/pip

COPY . .

CMD ["python", "training/train_cnn.py", "--config", "configs/cnn_default.yaml"]

# Licensed under the Apache License, Version 2.0 (2025)
# © Develop Aguascalientes & Copilot Microsoft
