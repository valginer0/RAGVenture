# RAGVenture Configuration Guide

This guide explains how to configure RAGVenture for optimal performance.

## Environment Variables

RAGVenture is designed to run completely free, with no API keys required! However, you can optionally use environment variables for custom configurations.

Create a `.env` file in your project root to configure RAGVenture:

```bash
# Required for text generation
HUGGINGFACE_TOKEN="your-token-here"  # Get from huggingface.co

# Optional: LangChain tracking (for debugging)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="your-langsmith-api-key"
LANGCHAIN_PROJECT="your-project-name"

# Optional: Market analysis configuration
MARKET_DATA_SOURCES="gartner,idc,crunchbase"
MARKET_ANALYSIS_REGION="north_america"
MARKET_CONFIDENCE_THRESHOLD="0.8"

# Optional: Model configuration
MODEL_TEMPERATURE="0.7"  # Controls creativity (0.0-1.0)
MODEL_MAX_LENGTH="1024"  # Maximum output length
MODEL_TOP_P="0.9"       # Nucleus sampling parameter
```

## Model Configuration

RAGVenture uses completely free, locally-running models by default:

### Text Generation Models

By default, RAGVenture uses GPT-2 which:
- Runs completely locally
- Requires no API key
- Has no usage costs
- Provides good performance for idea generation

You can configure the model behavior:

```python
from rag_startups.config import ModelConfig

# Default configuration
config = ModelConfig(
    model_name="gpt2",      # Free, locally-running model
    temperature=0.7,        # Higher = more creative, Lower = more focused
    max_length=1024,        # Maximum token length for generation
    top_p=0.9,             # Controls diversity of outputs
    num_return_sequences=1
)

# Example: More creative outputs
creative_config = ModelConfig(
    model_name="gpt2",
    temperature=0.9,
    top_p=0.95
)

# Example: More focused outputs
focused_config = ModelConfig(
    model_name="gpt2",
    temperature=0.5,
    top_p=0.8
)

# Optional: Use a different local model
config = ModelConfig(
    model_name="gpt2-medium",  # Larger model, still free
    temperature=0.7
)
```

### Embedding Models

We use Sentence Transformers for embeddings, which also:
- Run completely locally
- Require no API key
- Have no usage costs
- Provide excellent embedding quality

```python
from rag_startups.core.embeddings import get_embedding_model

# Default model (all-MiniLM-L6-v2)
embeddings = get_embedding_model()

# Custom model configuration
embeddings = get_embedding_model(
    model_name="all-mpnet-base-v2",  # Alternative model
    device="cuda"  # Use GPU if available
)
```

## CLI Configuration

Configure the CLI behavior through command-line arguments:

```bash
# Control idea generation creativity
python -m src.rag_startups.cli generate-all "AI" --temperature 0.8

# Adjust market analysis
python -m src.rag_startups.cli generate-all "fintech" --market

# Show relevant examples
python -m src.rag_startups.cli generate-all "edtech" --print-examples

# Use custom data source
python -m src.rag_startups.cli generate-all "biotech" --file "custom_data.json"
```

## Performance Optimization

### Memory Usage

Control memory usage through these settings:

```bash
# Optional: Memory optimization
CHUNK_SIZE="1000"           # Number of records to process at once
MAX_WORKERS="4"             # Number of parallel workers
CACHE_SIZE_MB="512"         # Maximum cache size in MB
```

### GPU Acceleration

Enable GPU acceleration if available:

```bash
# Optional: GPU configuration
USE_GPU="true"              # Enable GPU acceleration
CUDA_VISIBLE_DEVICES="0"    # Specify GPU device(s)
```

## Caching Configuration

Configure the caching system:

```python
from rag_startups.utils.cache import configure_cache

# Configure cache settings
configure_cache(
    backend="redis",           # 'redis' or 'local'
    ttl_seconds=3600,         # Cache expiry time
    max_size_mb=512,          # Maximum cache size
    compression=True          # Enable compression
)
```

## Logging Configuration

Control logging behavior:

```python
from rag_startups.utils.logging import configure_logging

# Configure logging
configure_logging(
    level="INFO",            # Logging level
    file="rag_startups.log", # Log file path
    format="detailed",       # Log format
    rotation="1 day"        # Log rotation
)
```

## Docker Configuration

When running in Docker, first make sure your `.env` file contains all necessary variables:

```bash
# Required
HUGGINGFACE_TOKEN="your-token-here"

# Optional
MODEL_TEMPERATURE=0.7
USE_GPU=true
MARKET_DATA_SOURCES=gartner,idc
CACHE_SIZE_MB=512
```

Then reference these variables in your docker-compose.yml:

```yaml
environment:
  # Required: Reference token from .env file
  - HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}

  # Optional: Reference other settings from .env
  - MODEL_TEMPERATURE=${MODEL_TEMPERATURE:-0.7}  # Default if not set
  - USE_GPU=${USE_GPU:-true}
  - MARKET_DATA_SOURCES=${MARKET_DATA_SOURCES:-gartner,idc}
  - CACHE_SIZE_MB=${CACHE_SIZE_MB:-512}
```

This way, all sensitive information like API tokens stays in your `.env` file (which should be in .gitignore), while docker-compose.yml can be safely committed to version control.

## Configuration Precedence

Configuration values are loaded in this order (later overrides earlier):

1. Default values in code
2. Environment variables
3. Configuration file settings
4. Command-line arguments

## Configuration File

You can also use a configuration file (config.yml):

```yaml
model:
  name: gpt2              # Free, locally-running model
  temperature: 0.7
  max_length: 1024
  top_p: 0.9

market_analysis:
  data_sources:
    - gartner
    - idc
    - crunchbase
  region: north_america
  confidence_threshold: 0.8

cache:
  backend: redis
  ttl_seconds: 3600
  max_size_mb: 512

logging:
  level: INFO
  file: rag_startups.log
  format: detailed
  rotation: 1 day
```

Load the configuration file:

```python
from rag_startups.config import load_config

config = load_config("config.yml")
