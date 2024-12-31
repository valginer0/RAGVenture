# RAGVenture API Reference

This document provides detailed information about RAGVenture's Python API.

## Command Line Interface

The main entry point is through the CLI module. Access it using:

```bash
python -m src.rag_startups.cli [command] [arguments]
```

### Commands

#### generate-all

Generate startup ideas with optional market analysis.

```bash
python -m src.rag_startups.cli generate-all "your topic" [options]
```

##### Required Arguments
- Topic or domain to generate startup ideas for (e.g., 'healthcare', 'education technology')

##### Optional Arguments
- `--num-ideas`: Number of ideas to generate (1-5, default: 1)
- `--file`: Path to startup data file (default: yc_startups.json)
- `--market/--no-market`: Include/exclude market analysis (default: include)
- `--temperature`: Model creativity (0.0-1.0, default: 0.7)
- `--print-examples`: Show relevant startup examples found in data

### Environment Variables

Required:
```bash
HUGGINGFACE_TOKEN="your-huggingface-token"  # Required for text generation
```

Optional LangSmith tracking:
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
python -m src.rag_startups.cli generate-all "healthcare" --num-ideas 5

# Use a different dataset with limited entries
python -m src.rag_startups.cli generate-all "education" --file "custom_startups.json" --max-lines 1000
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
