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
