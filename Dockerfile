# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set environment variables to force CPU-only installations
ENV FORCE_CUDA="0"
ENV TORCH_CUDA_ARCH_LIST="None"
ENV CUDA_VISIBLE_DEVICES=""

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project
COPY . .

# Install Python dependencies - force CPU versions
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    torchvision==0.16.0+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir -r requirements.txt

# Install and download spaCy model
RUN pip install --no-cache-dir spacy && \
    python -m spacy download en_core_web_sm

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:/app/src

# Create directories for logs and model cache
RUN mkdir -p /app/logs && chmod 777 /app/logs && \
    mkdir -p /app/.cache/models && chmod 777 /app/.cache/models && \
    mkdir -p /root/.cache/huggingface && chmod 777 /root/.cache/huggingface

# Set smart model management defaults
ENV RAG_SMART_MODELS=true
ENV RAG_MODEL_CHECK_INTERVAL=3600
ENV RAG_MODEL_TIMEOUT=60
