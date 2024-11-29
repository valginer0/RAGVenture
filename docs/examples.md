# RAGVenture Examples

> **⚠️ Important Note**: This document contains both tested and conceptual examples. The "Tested Examples" section includes functionality that has been thoroughly tested and is ready for production use. The "Conceptual Examples" section showcases potential use cases and patterns that are planned for future implementation. While these conceptual examples are designed based on the system's architecture, they haven't been fully tested yet. We're actively working on implementing and testing these features - contributions are welcome!

## 🧪 Tested Examples

These examples have corresponding test cases in our test suite and are ready for production use.

### 1. Basic Startup Analysis

```python
from rag_startups.core.rag_chain import initialize_rag, format_startup_idea
from rag_startups.core.startup_metadata import StartupLookup
from rag_startups.data.loader import load_data

# Load and initialize
df, json_data = load_data('data/startups.json')
lookup = StartupLookup()
for item in json_data:
    lookup.add_startup(item['long_desc'], item)
retriever = initialize_rag('data/startups.json')

# Analyze a startup idea
result = format_startup_idea(
    "A platform for connecting local farmers with restaurants",
    retriever,
    lookup
)
print(result)
```

### 2. Command Line Usage

```bash
# Generate single idea
python rag_startup_ideas.py --industry "Technology"

# Generate multiple ideas
python rag_startup_ideas.py --industry "Healthcare" --num-ideas 3
```

## 🔮 Conceptual Examples

> **Note**: The following examples demonstrate potential use cases and patterns. While they're designed to work with RAGVenture's architecture, they haven't been fully tested yet. We're working on implementing and testing these features in future releases.

### 1. Batch Processing (Planned)

```python
# Process multiple startup ideas
ideas = [
    "AI-powered personal fitness coach",
    "Sustainable packaging marketplace",
    "Remote team collaboration platform"
]

results = []
for idea in ideas:
    result = format_startup_idea(idea, retriever, lookup)
    results.append(result)
```

### 2. Custom Data Processing (Planned)

```python
import pandas as pd
from rag_startups.data.loader import load_data

# Load data with custom preprocessing
def custom_preprocess(df):
    # Convert year to datetime
    df['year'] = pd.to_datetime(df['year'], format='%Y')
    # Filter recent startups
    df = df[df['year'] >= '2020']
    return df

# Load and preprocess
df, json_data = load_data('data/startups.json')
df = custom_preprocess(df)

# Initialize with processed data
lookup = StartupLookup()
for _, row in df.iterrows():
    lookup.add_startup(row['long_desc'], row.to_dict())
```

### 3. Category-Specific Analysis (Planned)

```python
# Analyze startups in specific category
def analyze_category(description, category, retriever, lookup):
    result = format_startup_idea(description, retriever, lookup)
    
    # Filter similar companies by category
    similar = [
        company for company in result['Similar Companies']
        if company.get('category') == category
    ]
    
    result['Similar Companies'] = similar
    return result

# Example usage
tech_idea = "AI-powered code review tool"
tech_analysis = analyze_category(
    tech_idea,
    "Technology",
    retriever,
    lookup
)
```

### 4. Trend Analysis (Planned)

```python
from collections import Counter

# Analyze trends in startup categories
def analyze_trends(json_data, year=None):
    if year:
        data = [item for item in json_data if item['year'] == str(year)]
    else:
        data = json_data
        
    categories = Counter(item['category'] for item in data)
    return categories.most_common()

# Get trends for specific year
trends_2023 = analyze_trends(json_data, year=2023)
print("Top categories in 2023:", trends_2023)
```

### 5. Web Application Integration (Planned)

```python
from flask import Flask, request, jsonify
from rag_startups.core.rag_chain import initialize_rag, format_startup_idea

app = Flask(__name__)
retriever = initialize_rag('data/startups.json')
lookup = StartupLookup()

@app.route('/analyze', methods=['POST'])
def analyze_startup():
    data = request.json
    description = data.get('description')
    
    result = format_startup_idea(
        description,
        retriever,
        lookup
    )
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
```

### 6. Batch Processing Script (Planned)

```python
import csv
from pathlib import Path

def process_batch(input_file, output_file):
    # Load system
    retriever = initialize_rag('data/startups.json')
    lookup = StartupLookup()
    
    # Process ideas from CSV
    with open(input_file) as f_in, open(output_file, 'w') as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=['idea', 'analysis'])
        writer.writeheader()
        
        for row in reader:
            result = format_startup_idea(
                row['idea'],
                retriever,
                lookup
            )
            writer.writerow({
                'idea': row['idea'],
                'analysis': str(result)
            })

# Usage
process_batch('ideas.csv', 'analysis.csv')
```

## 🤝 Contributing New Examples

We welcome contributions! If you'd like to help implement and test any of the conceptual examples or add new ones:

1. Create tests in `tests/` directory
2. Implement the functionality
3. Update this documentation
4. Submit a pull request

See our [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed guidelines.