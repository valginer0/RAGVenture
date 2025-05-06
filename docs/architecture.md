# RAGVenture – Architecture & Design Overview

This document describes the **internal architecture, design principles, and AI-specific dependencies** that power the `rag_startups` (RAGVenture) project.

---

## 1. High-Level Architecture

```mermaid
flowchart TD
    A[Startup JSON / CSV Data] -->|pandas| B[DataFrame]
    B --> C[Document Objects]
    C -->|LangChain TextSplitter| D[Text Chunks]
    D -->|SentenceTransformer Embeddings| E[Vector Store – ChromaDB]
    F[User Prompt / Question] --> G[Retriever]
    E <-->|k-NN Search| G
    G --> H[Relevant Context]
    H & F -->|ChatPromptTemplate| I[Local LLM (transformers.pipeline)]
    I --> J[RAG-Formulated Answer]
```

* **Data Ingestion** – Raw YC (or custom) startup data is loaded with **`pandas`** and normalised into a DataFrame.
* **Chunking & Embedding** – Each startup description is split into manageable text chunks then converted to dense vectors using **Sentence-Transformers**.
* **Vector Store** – Embeddings are persisted locally in **ChromaDB**, enabling fast, semantic similarity search.
* **Retrieval** – A LangChain **retriever** performs k-NN search to fetch the most relevant chunks for a given user query.
* **Generation** – Retrieved context plus the user prompt are fed into a **local LLM** (via 🤗 **transformers.pipeline**) using a configurable **ChatPromptTemplate**. The model produces a structured startup idea.

---

## 2. Key Python Packages & Their Roles

| Package | Why it is Used | Where in Code |
|---------|----------------|---------------|
| **langchain** | Prompt templates, retriever abstraction, future agent integrations | `core/rag_chain.py`, `embeddings/`
| **chromadb** & **langchain-chroma** | Lightweight local vector store with persistence | `embeddings/embedding.py`
| **sentence-transformers** | State-of-the-art embeddings (`all-MiniLM-L6-v2` by default) | `embeddings/embedding.py`
| **transformers** | Runs the local text-generation model (e.g., `mistralai/Mistral-7B-Instruct`) via `pipeline("text-generation")` | `core/rag_chain.py`
| **pandas** | Efficient CSV/JSON parsing and manipulation | `data/loader.py`
| **spaCy** | Optional NLP for market/industry tagging | `analysis/market.py`
| **typer** | Rich CLI with autocompletion & help texts | `cli.py`
| **redis / fakeredis** | Optional in-memory cache for embeddings & lookups | `utils/cache.py`
| **backoff** | Robust retry logic when calling external APIs | `utils/retries.py`

> **All AI/ML workloads run *locally*** – there are *no* mandatory external API calls, keeping costs zero and data private.

---

## 3. Source Code Layout

```
rag_startups/
│
├─ src/rag_startups/
│   ├─ core/               # Retrieval-Augmented Generation logic
│   │   ├─ rag_chain.py    # ⬅ orchestrates retrieval + generation
│   │   └─ startup_metadata.py
│   ├─ embeddings/         # Embedding + vector-store helpers
│   ├─ data/               # Dataset loaders & converters
│   ├─ idea_generator/     # Domain-specific business logic
│   ├─ analysis/           # Market & competitive analysis (spaCy, wbdata)
│   ├─ utils/              # Timing, caching, exceptions
│   ├─ cli.py              # Typer command-line interface
│   └─ main.py             # Library entry-point
└─ docs/                   # Documentation (this file, user-guide, examples …)
```

### 3.1 `core/rag_chain.py`
The heart of the system. Important functions:

* `initialize_rag(df, json_data)` – Builds documents, splits them, stores embeddings, returns a **retriever** and a global **StartupLookup** helper.
* `rag_chain_local(question, generator, prompt_template, retriever)` – Executes the full RAG workflow to return the top-N formatted startup ideas.
* `format_startup_idea(...)` – Normalises raw LLM output into **Problem / Solution / Market / Value** sections.

### 3.2 Embedding Helpers
`embeddings/embedding.py` wraps creation of the vector store, hiding the underlying Chroma API. It also exposes `setup_retriever(store)` which is used by `core.rag_chain`.

### 3.3 CLI (`cli.py`)
Built with **Typer**, exposing commands such as `generate-all`. It spins up the RAG chain, handles flags (temperature, number of ideas, market analysis toggle), and prints rich coloured output to the terminal.

---

## 4. Design Principles

1. **Modularity** – Clear separation between data, embeddings, RAG core, and presentation layers.
2. **Local-First** – Runs fully offline; cloud APIs are *optional* (Hugging Face token for gated models, LangSmith for tracing).
3. **Speed via Caching** – Embeddings are persisted to disk and cached in Redis to avoid recomputation.
4. **Extensibility** – New data sources or models can be swapped by adding an adapter class without touching core logic.
5. **Observability** – Timing decorator records execution time; optional LangSmith tracing for deeper insights.
6. **Testability** – 60+ pytest unit tests ensure reliability; CI runs on every push.

---

## 5. Typical Execution Flow

1. **Load Data** – `StartupLoader` ingests `yc_startups.json` (or custom JSON).
2. **Start CLI** – `python -m src.rag_startups.cli generate-all "AI" --num-ideas 3`.
3. **Initialise RAG** – Vector store built (or loaded), retriever returned.
4. **User Prompt** – “AI security” passed to `rag_chain_local`.
5. **Retrieve Docs** – Top-K (default 5) similar YC startup descriptions fetched.
6. **Prompt Model** – Context + template fed to local LLM → startup idea text.
7. **Parse & Format** – Output structured into Problem/Solution/… sections.
8. **Display Result** – Printed to terminal or returned via API.

---

## 6. Extending the System

* **Swap Embedding Model**
  ```python
  from sentence_transformers import SentenceTransformer
  model = SentenceTransformer("intfloat/e5-small-v2")
  create_vectorstore(splits, embedding_model=model)
  ```
* **Add New Data Source** – Implement `CustomLoader` with `load()` → DataFrame and register in `data/__init__.py`.
* **Switch to GPU** – Pass `device_map="auto"` when instantiating the `pipeline`.

---

## 7. Future Architectural Enhancements

| Idea | Benefit |
|------|---------|
| **Agent-style Planner** | Multi-step market research chains |
| **Hybrid Search (BM25 + Vector)** | Improves recall on niche topics |
| **Distributed Embedding Cache (Faiss + GPU)** | Faster retrieval for 1M+ records |
| **Web UI (Streamlit)** | Non-technical user interface |

---

© 2025 RAGVenture – MIT License
