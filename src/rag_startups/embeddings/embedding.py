"""Embedding functionality for text processing."""
from typing import List, Union, Any

import numpy as np
from langchain_community.vectorstores.chroma import Chroma
from sentence_transformers import SentenceTransformer

from ..utils.exceptions import EmbeddingError
from ..utils.timing import timing_decorator
from config.config import DEFAULT_EMBEDDING_MODEL

class CustomEmbeddingFunction:
    """Custom embedding function using SentenceTransformer."""
    
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        """
        Initialize the embedding function.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            raise EmbeddingError(f"Failed to load embedding model: {e}")
    
    def embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        """
        Embed a list of documents.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings as numpy arrays
        """
        try:
            return [self.model.encode(text) for text in texts]
        except Exception as e:
            raise EmbeddingError(f"Failed to embed documents: {e}")
    
    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a single query text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding as numpy array
        """
        try:
            return self.model.encode(text)
        except Exception as e:
            raise EmbeddingError(f"Failed to embed query: {e}")

@timing_decorator
def create_vectorstore(
    texts: List[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL
) -> Chroma:
    """
    Create a vector store from texts using custom embeddings.
    
    Args:
        texts: List of texts to embed
        model_name: Name of the embedding model to use
        
    Returns:
        Chroma vector store
        
    Raises:
        EmbeddingError: If there's an error during embedding
    """
    try:
        custom_embedder = CustomEmbeddingFunction(model_name)
        return Chroma.from_texts(texts, embedding=custom_embedder)
    except Exception as e:
        raise EmbeddingError(f"Failed to create vector store: {e}")

@timing_decorator
def setup_retriever(vectorstore: Chroma) -> Union[None, Any]:
    """
    Set up a retriever from a vector store.
    
    Args:
        vectorstore: Chroma vector store
        
    Returns:
        Retriever object or None if setup fails
    """
    try:
        return vectorstore.as_retriever()
    except Exception as e:
        raise EmbeddingError(f"Failed to set up retriever: {e}")
