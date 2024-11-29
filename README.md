# RAGVenture
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![LangChain](https://img.shields.io/badge/powered%20by-LangChain-blue.svg)](https://github.com/hwchase17/langchain)
[![Sentence Transformers](https://img.shields.io/badge/powered%20by-Sentence%20Transformers-blue.svg)](https://www.sbert.net/)

RAGVenture is an intelligent startup analysis tool powered by Retrieval-Augmented Generation (RAG). It helps users explore, understand, and analyze startup companies by combining the power of large language models with precise information retrieval:

- 🎯 **Data-Driven**: Uses real startup data to ground its analysis and suggestions
- 🔍 **Context-Aware**: Understands and leverages patterns from successful startups
- 💡 **Intelligent**: Combines LLM capabilities with precise information retrieval
- 🆓 **Cost-Effective**: Runs entirely locally with no API costs

## 🎯 Why RAGVenture?

Traditional startup analysis tools either rely on expensive API calls or lack real-world context. RAGVenture solves this by:
1. Using local models (no API costs)
2. Grounding analysis in real startup data
3. Providing similar company analysis
4. Running entirely on your machine

## 🚀 Features

- **Smart Startup Analysis**: Analyze startup descriptions and extract key information using advanced RAG technology
- **Intelligent Metadata Extraction**: Automatically identify and extract company names, categories, and other metadata
- **Flexible Information Retrieval**: Find similar startups and related information using semantic search
- **Robust Data Processing**: Handle various data formats and edge cases with ease
- **Production-Ready Architecture**: Built with scalability and maintainability in mind
- **✨ Completely Free to Use**: 
  - Uses GPT-2 for text generation (no API key needed)
  - Sentence Transformers for embeddings (runs locally)
  - No external API dependencies or usage costs

## 🐳 Docker Support

The project provides Docker support for both CPU and GPU environments.

### Prerequisites

- Docker and Docker Compose installed
- For GPU support:
  - NVIDIA GPU with CUDA support
  - [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed
  - [nvidia-docker2](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) installed

### Running with Docker

#### CPU Version (Default)
For systems without GPU or when GPU acceleration is not needed:
```bash
docker-compose up app-cpu
```

#### GPU Version
For systems with NVIDIA GPU support:
```bash
docker-compose up app-gpu
```

The GPU version will automatically utilize your NVIDIA GPU for faster model inference.

### Building from Source with Docker

To build the images manually:

```bash
# Build CPU version
docker-compose build app-cpu

# Build GPU version (requires NVIDIA Container Toolkit)
docker-compose build app-gpu
```

### Docker Volumes

The application uses Docker volumes to cache models and embeddings:
- `model-cache`: Caches downloaded models
- `huggingface-cache`: Caches HuggingFace model files

These volumes persist between container restarts to avoid re-downloading models.

## 💻 System Requirements

- Python 3.11 or higher
- 8GB RAM minimum (16GB recommended)
- 2GB disk space for models and data
- Operating Systems:
  - Linux (recommended)
  - macOS
  - Windows (with WSL for best performance)

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/RAGVenture.git
cd RAGVenture

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🔧 Usage

The easiest way to analyze startup descriptions is using the script:

```bash
# Process a startup description
python rag_startup_ideas.py

# You can modify the question in the script:
# question = "Generate a company idea for the Real Estate based on provided context"
```

The script will:
1. Load startup data from `yc_startups.json`
2. Process your question using RAG
3. Generate a response based on similar startups

### Data Requirements

RAGVenture is designed to work with startup data in JSON format. The system was developed and tested using Y Combinator (YC) startup data, which is not included in this repository due to licensing considerations.

To use RAGVenture, you have two options:

1. **Recommended: Use YC Data** (Tested and Most Reliable): 
   
   a. Obtain YC Data:
   - Visit [Y Combinator's website](https://www.ycombinator.com/companies)
   - Use their "Download as CSV" feature (requires login)
   - Convert the CSV to JSON using our conversion script:
     ```bash
     # Convert YC data to JSON
     python -m rag_startups.data.convert_yc_data path/to/yc_data.csv -o startups.json
     ```
   
   The conversion script handles:
   - Cleaning and formatting descriptions
   - Extracting years from batch information
   - Removing entries without names or descriptions
   - Proper UTF-8 encoding
   
   Expected CSV format from YC:
   ```csv
   Company Name,Description,long_desc,Category,Batch
   Airbnb,Marketplace for unique accommodations...,Marketplace,S08
   ```

   b. Example YC Data Format:
   ```json
   [
     {
       "name": "Airbnb",
       "description": "Marketplace for unique accommodations",
       "long_desc": "Founded in August of 2008 and based in San Francisco, California, Airbnb is a trusted community marketplace for people to list, discover, and book unique accommodations around the world. The platform connects hosts and travelers, offering unique spaces for memorable experiences.",
       "category": "Marketplace",
       "year": "2008"
     },
     {
       "name": "Dropbox",
       "description": "Cloud storage and synchronization",
       "long_desc": "Dropbox is a file hosting service that offers cloud storage, file synchronization, and client software. The service enables users to store and share files across devices, with features for both individual users and enterprise customers.",
       "category": "Enterprise Software",
       "year": "2007"
     }
   ]
   ```

2. **Alternative: Use Your Own Data**: 
   If you prefer to use your own dataset, prepare it in the following format shown above.

For testing purposes, we provide a minimal sample dataset in `data/sample_startups.json` with a few fictional entries, but for best results, we recommend using the YC dataset.

### Example Output

```bash
$ python rag_startup_ideas.py

{
    "name": "RentEase",
    "description": "A platform for renting homes and apartments globally",
    "long_desc": "RentEase is a comprehensive rental platform that connects property owners with potential tenants worldwide. The platform features advanced search capabilities, secure payment processing, virtual tours, and a review system to build trust in the community. Property owners can easily list their spaces while renters can find their ideal accommodations.",
    "category": "Real Estate",
    "year": "2024",
    "similar_companies": [
        {
            "name": "Airbnb",
            "description": "Trusted marketplace for unique accommodations worldwide",
            "similarity_score": 0.89
        },
        {
            "name": "VRBO",
            "description": "Vacation rental marketplace connecting homeowners and travelers",
            "similarity_score": 0.82
        },
        {
            "name": "Zillow",
            "description": "Real estate and rental marketplace platform",
            "similarity_score": 0.75
        }
    ]
}
```

Using the Python API:
```python
from rag_startups.core.rag_chain import initialize_rag, format_startup_idea
from rag_startups.core.startup_metadata import StartupLookup

# Load startup data and initialize the system
data = load_startup_data('data/startups.json')  # Load your startup data
lookup = StartupLookup()
for item in data:
    lookup.add_startup(item['long_desc'], item)  # Use long_desc for RAG processing

# Initialize RAG system
retriever = initialize_rag('data/startups.json')

# Analyze a startup
result = format_startup_idea(
    "An AI-powered customer support platform",
    retriever,
    lookup
)
print(result)
```

### Python API

For more advanced usage, you can use the Python API:

```python
from rag_startups.core.rag_chain import initialize_rag, format_startup_idea
from rag_startups.core.startup_metadata import StartupLookup

# Load startup data and initialize the system
data = load_startup_data('data/startups.json')  # Load your startup data
lookup = StartupLookup()
for item in data:
    lookup.add_startup(item['long_desc'], item)  # Use long_desc for RAG processing

# Initialize RAG system
retriever = initialize_rag('data/startups.json')

# Analyze a startup
result = format_startup_idea(
    "An AI-powered customer support platform",
    retriever,
    lookup
)
print(result)
```

## 🏗️ Architecture

RAGVenture uses a modular architecture:

- `core/`: Core RAG functionality and startup analysis
- `data/`: Data loading and processing utilities
- `embeddings/`: Vector embeddings and similarity search
- `utils/`: Helper functions and utilities

## 📚 Documentation

For detailed documentation, see:
- [API Reference](docs/api.md)
- [Configuration Guide](docs/configuration.md)
- [Advanced Usage Examples](docs/examples.md)

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_real_data.py
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:
- Code of Conduct
- Development setup
- Submission guidelines
- Code style
- Testing requirements

## ❓ Troubleshooting

### Common Issues

1. **ModuleNotFoundError**:
   ```
   ModuleNotFoundError: No module named 'transformers'
   ```
   Solution: Ensure you've installed all requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. **CUDA/GPU Issues**:
   - By default, RAGVenture runs on CPU
   - For GPU support, install PyTorch with CUDA (see PyTorch website)

3. **Memory Issues**:
   - Reduce batch size in config
   - Use smaller model variants
   - Process data in smaller chunks

For more issues, please check our [Issues](https://github.com/valginer0/rag_startups/issues) page.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [LangChain](https://github.com/hwchase17/langchain)
- Powered by [Sentence Transformers](https://www.sbert.net/)
- Inspired by Y Combinator startups