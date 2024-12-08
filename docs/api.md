# RAGVenture API Reference

This document provides detailed information about RAGVenture's Python API.

## Main Script

The main entry point is `rag_startup_ideas.py` in the root directory. It provides a command-line interface for generating startup ideas using RAG technology.

### Command Line Usage

```bash
python rag_startup_ideas.py --topic "your topic" [options]
```

#### Required Arguments
- `--topic`: Topic or domain to generate startup ideas for (e.g., 'healthcare', 'education technology')

#### Optional Arguments
- `--file`: Path to the JSON file containing startup data (default: yc_startups.json)
- `--max-lines`: Maximum number of lines to process
- `--num-ideas`: Number of startup ideas to generate (default: 3)

### Environment Variables

To use LangSmith tracking, set up these environment variables:
```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="your lansmith api key"
LANGCHAIN_PROJECT="the name of your langsmith project"
```

## Core Components

### StartupLookup

::: rag_startups.core.startup_metadata.StartupLookup
    handler: python
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

### RAG Chain

```python
from rag_startups.core.rag_chain import initialize_rag, format_startup_idea
```

#### Functions

- `initialize_rag(file_path: str) -> BaseRetriever`
  - Initializes the RAG system with startup data
  - Parameters:
    - `file_path`: Path to JSON file containing startup data
  - Returns: Configured retriever for similarity search

- `format_startup_idea(description: str, retriever: BaseRetriever, lookup: StartupLookup) -> dict`
  - Analyzes a startup description and returns formatted results
  - Parameters:
    - `description`: Startup idea to analyze
    - `retriever`: Initialized retriever from `initialize_rag()`
    - `lookup`: Initialized StartupLookup instance
  - Returns: Dictionary containing analysis results

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
lookup = StartupLookup(json_data)

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

### Command Line Example

```bash
# Generate 5 healthcare startup ideas
python rag_startup_ideas.py --topic "healthcare" --num-ideas 5

# Use a different dataset with limited entries
python rag_startup_ideas.py --topic "education" --file "custom_startups.json" --max-lines 1000
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

```python
# Removed reference to non-existent file_utils module
