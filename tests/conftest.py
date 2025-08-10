from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _mock_hf_model_info_autouse():
    """Mock huggingface_hub.model_info used by CLI preflight to avoid network/auth.
    Applies to all tests importing rag_startups.cli.
    """
    with patch("rag_startups.cli.model_info") as mi:
        mi.return_value = object()
        yield mi


@pytest.fixture(autouse=True)
def _ensure_hf_token_env(monkeypatch):
    """Ensure HF token env vars exist so RAGSettings validation passes in CI."""
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "test_token")
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "test_token")
    monkeypatch.setenv("HF_TOKEN", "test_token")


@pytest.fixture(autouse=True)
def _mock_initialize_embeddings(monkeypatch):
    """Avoid real SentenceTransformer/Chroma by returning a lightweight retriever.

    This prevents network/downloads and speeds up CLI-related tests that call
    initialize_embeddings() via find_relevant_startups().
    """

    class _FakeDoc:
        def __init__(self, text: str):
            self.page_content = text

    class _FakeRetriever:
        def invoke(self, query: str):
            # Return a couple of simple docs with page_content, enough for downstream code
            return [
                _FakeDoc("Example Startup A: AI tools for data cleaning."),
                _FakeDoc("Example Startup B: Analytics platform for SMBs."),
            ]

    # Replace initialize_embeddings with a function returning our fake retriever
    monkeypatch.setattr(
        "rag_startups.embed_master.initialize_embeddings",
        lambda df, model_name="all-MiniLM-L6-v2": _FakeRetriever(),
        raising=True,
    )


@pytest.fixture(autouse=True)
def _mock_sentence_transformer(monkeypatch):
    """Mock SentenceTransformer to avoid any HF network/auth in tests.

    Many tests import paths that instantiate SentenceTransformer directly
    (e.g., in rag chain utilities). Replacing it with a simple fake ensures
    no downloads and prevents 401 Unauthorized in CI when an invalid token
    is present.
    """

    class _FakeST:
        def __init__(self, model_name: str = "all-MiniLM-L6-v2", **kwargs):
            self.model_name = model_name

        def encode(self, sentences, **kwargs):
            # Return deterministic vectors with expected dimensionality (384)
            import numpy as np

            if isinstance(sentences, str):
                rng = np.random.default_rng(42)
                return rng.random(384, dtype=float)
            else:
                n = len(sentences)
                rng = np.random.default_rng(42)
                return rng.random((n, 384), dtype=float)

    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer",
        _FakeST,
        raising=True,
    )
