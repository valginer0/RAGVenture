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
command: python -m src.rag_startups.cli generate-all --num-ideas 3 "fintech"
```

Available command options:
- `generate-all`: Generate startup ideas for a given topic
  - Required: topic (e.g., "fintech", "AI/ML", "education")
  - `--num-ideas`: Number of ideas to generate (1-5, default: 1)
  - `--file`: Path to startup data file (default: yc_startups.json)
  - `--market/--no-market`: Include/exclude market analysis (default: include)
  - `--temperature`: Model creativity (0.0-1.0, default: 0.7)
  - `--print-examples`: Show relevant startup examples found

## Volumes

The setup includes persistent volumes for:
- Model cache (sentence transformers)
- Hugging Face cache
- Log files
