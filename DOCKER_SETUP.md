# Docker Setup Guide

## Prerequisites

1. Docker and Docker Compose installed
2. HuggingFace API token
3. BLS API key (for market analysis)
4. YC Startups data file (must be downloaded separately)

## Setup Steps

1. Create required directories:
   ```bash
   mkdir -p data logs
   ```

2. Place your startup data file in the data directory:
   ```bash
   # Copy your yc_startups.json to the data directory
   cp path/to/your/yc_startups.json data/
   ```

3. Create .env file with your API keys:
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys:
   # HUGGINGFACE_TOKEN=your_token_here
   # BLS_API_KEY=your_bls_key_here
   ```

4. Build and run:
   ```bash
   # CPU version
   docker-compose up app-cpu

   # GPU version (requires NVIDIA Docker)
   docker-compose up app-gpu
   ```

## Directory Structure

```
rag_startups/
├── data/               # Mount point for startup data
│   └── yc_startups.json
├── logs/              # Mount point for application logs
├── src/               # Application source code
├── tests/             # Test files
├── config/            # Configuration files
├── .env              # Environment variables (create from .env.example)
├── .env.example      # Example environment file
├── docker-compose.yml
└── Dockerfile
```

## Required Files

1. `data/yc_startups.json`: Your startup data file (must be provided)
2. `.env`: Environment file with API keys
3. `logs/`: Directory for application logs

## Configuration

You can customize the behavior by:

1. Modifying environment variables in `.env`
2. Changing command arguments in `docker-compose.yml`

Example command arguments:
```yaml
command: >
  python -m rag_startups.cli generate
  --topic "AI/ML"          # Required: Topic for idea generation
  --file /app/data/yc_startups.json  # Optional: Path to startup data
  --num 3                  # Optional: Number of ideas (1-5)
  --market                 # Optional: Include market analysis
```

## Notes

- The startup data file must be provided by the user
- API keys must be set in the .env file
- Logs are persisted in the ./logs directory
- Model caches are persisted in Docker volumes
