"""Custom exceptions for the RAG Startups project."""

class RAGStartupsError(Exception):
    """Base exception for RAG Startups project."""
    pass

class DataLoadError(RAGStartupsError):
    """Raised when there is an error loading data."""
    pass

class EmbeddingError(RAGStartupsError):
    """Raised when there is an error during embedding."""
    pass

class ModelError(RAGStartupsError):
    """Raised when there is an error with the language model."""
    pass

class ConfigError(RAGStartupsError):
    """Raised when there is a configuration error."""
    pass
