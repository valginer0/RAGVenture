# RAGVenture Configuration Guide

This guide explains how to configure RAGVenture for optimal performance with the new smart model management system.

## Smart Model Management

RAGVenture now includes intelligent model management that automatically handles model failures, deprecation, and provides local fallback options.

### Key Features
- **Automatic Fallback**: Falls back to local models when external APIs fail
- **Model Migration**: Handles model version updates automatically (e.g., Mistral v0.2→v0.3)
- **Health Monitoring**: Continuous model health checks
- **Local Resilience**: Works completely offline

## Environment Variables

RAGVenture is designed to run completely FREE, with no API keys required! The system works entirely offline with local models by default. All configuration is optional for enhanced features.

Create a `.env` file in your project root to configure RAGVenture:

```bash
# Optional: Add your HuggingFace token for enhanced remote models
# (system works completely FREE without it)
# HUGGINGFACE_TOKEN=your_actual_token_here

# Smart Model Management (enabled by default)
RAG_SMART_MODELS=true                    # Enable intelligent model selection
RAG_MODEL_CHECK_INTERVAL=3600            # Health check interval (seconds)
RAG_MODEL_TIMEOUT=60                     # Model response timeout (seconds)

# Language Model Configuration (uses smart model management)
RAG_LANGUAGE_MODEL="mistralai/Mistral-7B-Instruct-v0.3"  # Preferred language model
RAG_EMBEDDING_MODEL="all-MiniLM-L6-v2"                   # Default embedding model
RAG_MODEL_TEMPERATURE=0.7                                # Model creativity (0.0-1.0)

# Optional: LangChain tracking (for debugging)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="your-langsmith-api-key"
LANGCHAIN_PROJECT="your-project-name"

# Optional: Market analysis configuration
MARKET_DATA_SOURCES="gartner,idc,crunchbase"
MARKET_ANALYSIS_REGION="north_america"
MARKET_CONFIDENCE_THRESHOLD="0.8"
```

## Model Configuration

RAGVenture uses smart model management with automatic fallback:

### Smart Model Selection

The system automatically selects the best available model:
1. **Remote Models**: Mistral-7B-Instruct-v0.3 (if HuggingFace token available)
2. **Local Fallback**: GPT-2 and other local models
3. **Migration Intelligence**: Automatically handles model deprecation

### Model Management CLI

```bash
# Check model health and status
python -m rag_startups.cli models status

# List available models with priorities
python -m rag_startups.cli models list

# Test specific models
python -m rag_startups.cli models test mistralai/Mistral-7B-Instruct-v0.3
```

### Programmatic Configuration

```python
from rag_startups.config.settings import Settings

# Load configuration with smart model management
settings = Settings(
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

RAGVenture works completely FREE in Docker with no configuration required! For enhanced features, create a `.env` file:

```bash
# Optional: Add your tokens for enhanced features (system works FREE without them)
# HUGGINGFACE_TOKEN=your_actual_token_here
# BLS_API_KEY=your_bls_key_here

# Smart Model Management (enabled by default in Docker)
RAG_SMART_MODELS=true
RAG_MODEL_CHECK_INTERVAL=3600
RAG_MODEL_TIMEOUT=60

# Optional: Model preferences
RAG_MODEL_TEMPERATURE=0.7
RAG_LANGUAGE_MODEL=mistralai/Mistral-7B-Instruct-v0.3
```

The docker-compose.yml automatically references these variables:

```yaml
environment:
  # Smart model management (enabled by default)
  - RAG_SMART_MODELS=${RAG_SMART_MODELS:-true}
  - RAG_MODEL_CHECK_INTERVAL=${RAG_MODEL_CHECK_INTERVAL:-3600}
  - RAG_MODEL_TIMEOUT=${RAG_MODEL_TIMEOUT:-60}

  # Optional: Enhanced features (system works FREE without these)
  - HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}
  - RAG_MODEL_TEMPERATURE=${RAG_MODEL_TEMPERATURE:-0.7}
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
