# RAGVenture

RAGVenture is an intelligent startup analysis tool powered by Retrieval-Augmented Generation (RAG). It helps users explore, understand, and analyze startup companies by combining the power of large language models with precise information retrieval.

## 🚀 Features

- **Smart Startup Analysis**: Analyze startup descriptions and extract key information using advanced RAG technology
- **Intelligent Metadata Extraction**: Automatically identify and extract company names, categories, and other metadata
- **Flexible Information Retrieval**: Find similar startups and related information using semantic search
- **Robust Data Processing**: Handle various data formats and edge cases with grace
- **Production-Ready Architecture**: Built with scalability and maintainability in mind

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
   Company Name,Description,Category,Batch
   Airbnb,Marketplace for unique accommodations...,Marketplace,S08
   ```

   b. Example YC Data Format:
   ```json
   [
     {
       "name": "Airbnb",
       "description": "Founded in August of 2008 and based in San Francisco, California, Airbnb is a trusted community marketplace for people to list, discover, and book unique accommodations around the world.",
       "category": "Marketplace",
       "year": "2008"
     },
     {
       "name": "Dropbox",
       "description": "Dropbox is a file hosting service that offers cloud storage, file synchronization, and client software.",
       "category": "Enterprise Software",
       "year": "2007"
     }
   ]
   ```

2. **Alternative: Use Your Own Data**: 
   If you prefer to use your own dataset, prepare it in the following format shown above.

For testing purposes, we provide a minimal sample dataset in `data/sample_startups.json` with a few fictional entries, but for best results, we recommend using the YC dataset.

### Example Output

When you run RAGVenture with YC data, here's what you get:

```bash
$ python rag_startup_ideas.py

{
    "Description": "A platform for renting homes and apartments globally",
    "Company": "Airbnb",
    "Category": "Marketplace",
    "Year": "2008",
    "Similar Companies": [
        "Airbnb: Trusted marketplace for unique accommodations worldwide",
        "VRBO: Vacation rental marketplace connecting homeowners and travelers",
        "Zillow: Real estate and rental marketplace platform"
    ]
}
```

Using the Python API:
```python
result = format_startup_idea(
    "A platform for renting homes and apartments globally",
    retriever,
    lookup
)
print(result)
# Output:
# {
#     "Description": "A platform for renting homes and apartments globally",
#     "Company": "Airbnb",
#     "Category": "Marketplace",
#     "Year": "2008"
# }
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
    lookup.add_startup(item['description'], item)

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

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [LangChain](https://github.com/langchain-ai/langchain)
- Uses [ChromaDB](https://github.com/chroma-core/chroma) for vector storage
- Powered by Sentence Transformers for embeddings