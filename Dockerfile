# Dockerfile

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The command to run will be specified in docker-compose.yml