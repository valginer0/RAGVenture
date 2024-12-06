# RAGVenture
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![LangChain](https://img.shields.io/badge/powered%20by-LangChain-blue.svg)](https://github.com/hwchase17/langchain)
[![Sentence Transformers](https://img.shields.io/badge/powered%20by-Sentence%20Transformers-blue.svg)](https://www.sbert.net/)
[![CI](https://github.com/valginer0/rag_startups/actions/workflows/ci.yml/badge.svg)](https://github.com/valginer0/rag_startups/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/valginer0/rag_startups/graph/badge.svg)](https://codecov.io/gh/valginer0/rag_startups)

RAGVenture is an intelligent startup idea generator powered by Retrieval-Augmented Generation (RAG). It helps entrepreneurs generate innovative startup ideas by learning from successful companies, combining the power of large language models with real-world startup data.

## Why RAGVenture?

Traditional startup ideation tools either rely on expensive API calls or generate ideas without real-world context. RAGVenture solves this by:
- Data-Driven: Learns from real startup data to ground suggestions in reality
- Context-Aware: Understands patterns from successful startups
- Intelligent: Uses RAG to combine LLM capabilities with precise information retrieval
- Fast & Local: Runs entirely on your machine with no API costs (except optional Hugging Face)
- Production-Ready: Comprehensive test suite, error handling, and monitoring

## System Requirements

- Python 3.11 or higher
- 8GB RAM minimum (16GB recommended)
- 2GB disk space for models and data
- Operating Systems:
  - Linux (recommended)
  - macOS
  - Windows (with WSL for best performance)

## Quick Start

1. **Installation**:
```bash
# Clone the repository
git clone https://github.com/yourusername/RAGVenture.git
cd RAGVenture

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Unix or MacOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

2. **Environment Setup**:
```bash
# Required for text generation
export HUGGINGFACE_TOKEN="your-token-here"  # Get from huggingface.co

# Optional for LangChain tracing (debugging)
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
export LANGCHAIN_API_KEY="your-langsmith-api-key"
export LANGCHAIN_PROJECT="your-project-name"
```

3. **Generate Ideas**:
```bash
python rag_startup_ideas.py --topic "AI" --num-ideas 1
```

## Features & Capabilities

### Core Features
- Intelligent Idea Generation:
  - Uses RAG to combine LLM knowledge with real startup data
  - Generates contextually relevant and grounded ideas
  - Provides structured output with problem, solution, and market analysis

### Command-Line Interface
- --topic: Target domain or area (required)
- --num-ideas: Number of ideas to generate (default: 3)
- --file: Custom startup data file (default: yc_startups.json)
- --max-lines: Limit data processing (optional)

### Technical Features
- Smart Analysis:
  - Semantic search for relevant examples
  - Automatic metadata extraction
  - Pattern recognition from successful startups
- Performance Optimized:
  - One-time embedding generation (~22s)
  - Fast idea generation (~0.5s per idea)
  - Efficient data processing (~0.1s load time)
- Production Quality:
  - 31 comprehensive unit tests
  - Automated code formatting
  - Extensive error handling

## Performance

Typical processing times on a standard machine:
- Initial Setup: ~22s (one-time embedding generation)
- Data Loading: ~0.1s
- Idea Generation: ~0.5s per idea

## Docker Support

For containerized deployment, we provide both CPU and GPU support.

### Prerequisites
- Docker and Docker Compose
- For GPU support:
  - NVIDIA GPU with CUDA
  - NVIDIA Container Toolkit
  - nvidia-docker2

### Quick Start with Docker
```bash
# CPU Version
docker-compose up app-cpu

# GPU Version (with NVIDIA support)
docker-compose up app-gpu
```

## Development Setup

1. Clone and setup:
```bash
git clone https://github.com/yourusername/RAGVenture.git
cd RAGVenture
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install development dependencies:
```bash
pip install -r requirements.txt
pre-commit install  # Sets up automatic code formatting
```

3. Run tests:
```bash
pytest tests/  # Should show 31 passing tests
```

## Data Requirements

RAGVenture works with startup data in JSON format. Two options:

1. Use YC Data (Recommended):
   - Download from [Y Combinator](https://www.ycombinator.com/companies)
   - Convert CSV to JSON:
     ```bash
     python -m rag_startups.data.convert_yc_data input.csv -o startups.json
     ```

2. Use Custom Data:
   - Prepare JSON file with required fields
   - See `docs/data_format.md` for schema

## Troubleshooting

1. Embedding Generation Time:
   - First run takes ~22s to generate embeddings
   - Subsequent runs use cached embeddings
   - GPU can significantly speed up this process

2. Common Issues:
   - Missing HUGGINGFACE_TOKEN: Sign up at huggingface.co
   - Memory errors: Reduce batch size with --max-lines
   - GPU errors: Ensure CUDA toolkit is properly installed

## Documentation

- `docs/api.md`: API documentation
- `docs/examples.md`: Usage examples
- `docs/data_format.md`: Data schema
- `CONTRIBUTING.md`: Development guidelines

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.
