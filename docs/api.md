# RAGVenture API Reference

This document provides detailed information about RAGVenture's Python API.

## Core Components

### StartupLookup

::: rag_startups.core.startup_metadata.StartupLookup
    handler: python
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

### RAG Components

::: rag_startups.rag_startup_ideas
    handler: python
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

### Data Loading

```python
from rag_startups.data.loader import load_data
```

#### Functions

- `load_data(file_path: str, max_lines: Optional[int] = None) -> Tuple[pd.DataFrame, List[dict]]`
  - Loads startup data from JSON file
  - Parameters:
    - `file_path`: Path to JSON file
    - `max_lines`: Optional limit on number of entries to load
  - Returns: Tuple of (DataFrame, raw JSON data)

## Usage Examples

### Basic Usage

```python
from rag_startups.core.rag_chain import initialize_rag, format_startup_idea
from rag_startups.core.startup_metadata import StartupLookup
from rag_startups.data.loader import load_data

# Load data
df, json_data = load_data('data/startups.json')

# Initialize lookup
lookup = StartupLookup()
for item in json_data:
    lookup.add_startup(item['description'], item)

# Initialize RAG
retriever = initialize_rag('data/startups.json')

# Analyze startup
result = format_startup_idea(
    "An AI-powered customer support platform",
    retriever,
    lookup
)
print(result)
```

### Custom Configuration

```python
from rag_startups.core.rag_chain import initialize_rag
from langchain.embeddings import HuggingFaceEmbeddings

# Use custom embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cuda'}  # Use GPU if available
)

# Initialize with custom configuration
retriever = initialize_rag(
    'data/startups.json',
    embeddings_model=embeddings
)
```

## Error Handling

The API uses Python's built-in exception handling. Common exceptions include:

- `FileNotFoundError`: When startup data file is not found
- `JSONDecodeError`: When startup data file contains invalid JSON
- `ValueError`: When input parameters are invalid

Example error handling:

```python
try:
    df, json_data = load_data('data/startups.json')
except FileNotFoundError:
    print("Startup data file not found")
except json.JSONDecodeError:
    print("Invalid JSON format in startup data file")
```

## Utility Functions

::: rag_startups.utils
    handler: python
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3
