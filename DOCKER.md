# Docker Setup Guide

This guide explains how to run the RAG Startups project using Docker.

## Prerequisites

1. Docker and Docker Compose installed on your system
2. A HuggingFace API token (optional - system works with local models)
3. The `yc_startups.json` data file
4. Optional: BLS API key (for enhanced market analysis features)

## Quick Start

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your configuration:
   ```
   # Optional: HuggingFace token for remote models (system works without it)
   HUGGINGFACE_TOKEN=your_token_here

   # Smart model management (enabled by default)
   RAG_SMART_MODELS=true
   RAG_MODEL_CHECK_INTERVAL=3600
   RAG_MODEL_TIMEOUT=30

   # Optional: Enhanced market analysis
   BLS_API_KEY=your_bls_key_here
   ```

3. Place your `yc_startups.json` file in the project root directory.

4. Run with Docker Compose:
   ```bash
   # For CPU version (recommended):
   docker-compose up app-cpu

   # For GPU version (requires NVIDIA Docker):
   docker-compose up app-gpu

   # Check model health and status:
   docker-compose up model-manager
   ```

## Configuration

You can customize the behavior by:

1. Modifying environment variables in `.env`
2. Changing the command arguments in `docker-compose.yml`

Example command arguments:
```yaml
command: python -m rag_startups.cli generate-all --num-ideas 3 "fintech"
```

## Smart Model Management

RAGVenture includes intelligent model management that automatically handles:

### Automatic Fallback
- **Remote Model Failures**: Automatically falls back to local models when HuggingFace models are unavailable
- **Model Deprecation**: Handles model version updates (e.g., Mistral v0.2 → v0.3) automatically
- **Network Issues**: Continues working offline with local models

### Model Health Monitoring
```bash
# Check model status in Docker
docker-compose up model-manager

# Or run model commands directly
docker-compose run app-cpu python -m rag_startups.cli models status
docker-compose run app-cpu python -m rag_startups.cli models list
```

### Environment Variables for Model Management
```yaml
environment:
  - RAG_SMART_MODELS=true              # Enable smart model selection
  - RAG_MODEL_CHECK_INTERVAL=3600      # Health check interval (seconds)
  - RAG_MODEL_TIMEOUT=30               # Model response timeout (seconds)
```

### Persistent Model Cache
The Docker setup includes persistent volumes for model caching:
- `model-cache`: Sentence transformer models
- `huggingface-cache`: HuggingFace model cache
- `smart-model-cache`: Smart model management cache

This ensures models are downloaded once and reused across container restarts.

Available command options:
- `generate-all`: Generate startup ideas for a given topic
  - Required: topic (e.g., "fintech", "AI/ML", "education")
- `models`: Smart model management commands
  - `status`: Check health and status of all models
  - `list`: List available models with priorities
  - `test`: Test specific models for availability
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
