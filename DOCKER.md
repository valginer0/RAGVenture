# Docker Setup Guide

This guide explains how to run the RAG Startups project using Docker.

## Prerequisites

1. Docker and Docker Compose installed on your system
2. A HuggingFace API token
3. A BLS API key (for market analysis features)
4. The `yc_startups.json` data file

## Quick Start

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your API keys:
   ```
   HUGGINGFACE_TOKEN=your_token_here
   BLS_API_KEY=your_bls_key_here
   ```

3. Place your `yc_startups.json` file in the project root directory.

4. Run with Docker Compose:
   ```bash
   # For CPU version:
   docker-compose up app-cpu

   # For GPU version (requires NVIDIA Docker):
   docker-compose up app-gpu
   ```

## Configuration

You can customize the behavior by:

1. Modifying environment variables in `.env`
2. Changing the command arguments in `docker-compose.yml`

Example command arguments:
```yaml
command: python -m rag_startups.cli --topic "AI/ML" --num-ideas 5
```

## Volumes

The setup includes persistent volumes for:
- Model cache (sentence transformers)
- HuggingFace cache

This ensures faster subsequent runs by caching downloaded models.
