# RAGVenture Configuration Guide

This guide explains how to configure RAGVenture for optimal performance.

## Environment Variables

RAGVenture is designed to run completely free, with no API keys required! However, you can optionally use environment variables for custom configurations. Create a `.env` file in your project root:

```bash
# Optional: LangChain tracking (for debugging)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"

# Optional: Only needed if you want to use OpenAI models instead of the default free GPT-2
OPENAI_API_KEY="your-openai-key"
```

## Model Configuration

RAGVenture uses completely free, locally-running models by default:

### Text Generation Model

By default, we use GPT-2 which:
- Runs completely locally
- Requires no API key
- Has no usage costs
- Provides good performance for idea generation

```python
# Default configuration (no action needed)
from settings import local_language_model_name  # This is 'gpt2'

# Optional: Use a different local model
local_language_model_name = 'gpt2-medium'  # Larger model, still free
```

### Embedding Models

We use Sentence Transformers by default, which also:
- Run completely locally
- Require no API key
- Have no usage costs
- Provide excellent embedding quality

1. Default Model (all-MiniLM-L6-v2):
```python
from rag_startups.core.rag_chain import initialize_rag

# Uses free Sentence Transformers model
retriever = initialize_rag('data/startups.json')
```

2. Optional: Custom Model:
```python
from langchain.embeddings import HuggingFaceEmbeddings

# Still free, just a different model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
retriever = initialize_rag('data/startups.json', embeddings_model=embeddings)
```

### Vector Store Configuration

RAGVenture uses Chroma as the vector store. Configure it in `initialize_rag`:

```python
from langchain.vectorstores import Chroma

retriever = initialize_rag(
    'data/startups.json',
    collection_name="my_startups",  # Custom collection name
    persist_directory="./chroma_db"  # Where to store vectors
)
```

## Performance Tuning

### Memory Usage

Control memory usage by limiting data loading:

```python
from rag_startups.data.loader import load_data

# Load only first 1000 startups
df, json_data = load_data('data/startups.json', max_lines=1000)
```

### Processing Speed

Optimize for speed with batch processing:

```python
from rag_startups.core.startup_metadata import StartupLookup

# Batch process startups
lookup = StartupLookup()
batch_size = 100
for i in range(0, len(json_data), batch_size):
    batch = json_data[i:i+batch_size]
    for item in batch:
        lookup.add_startup(item['description'], item)
```

## Data Format

### Input JSON Schema

Your startup data should follow this schema:

```json
{
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "name": {
        "type": "string",
        "description": "Company name"
      },
      "description": {
        "type": "string",
        "description": "Brief company description"
      },
      "long_desc": {
        "type": "string",
        "description": "Detailed company description used for RAG processing"
      },
      "category": {
        "type": "string",
        "description": "Business category"
      },
      "year": {
        "type": "string",
        "description": "Founding year"
      }
    },
    "required": ["name", "description", "long_desc"]
  }
}
```

### Custom Data Fields

Add custom fields to startup metadata:

```python
# Add custom fields
startup_data = {
    "name": "TechCorp",
    "description": "AI platform",
    "long_desc": "TechCorp is an innovative AI platform that leverages cutting-edge machine learning algorithms to solve complex business problems. The system provides automated decision-making capabilities, predictive analytics, and real-time optimization across various business domains.",
    "category": "Technology",
    "year": "2023",
    "custom_field": "value"  # Custom field
}
lookup.add_startup(startup_data['long_desc'], startup_data)
