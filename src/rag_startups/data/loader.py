"""Data loading and processing functionality."""
from pathlib import Path
from typing import Optional

import pandas as pd
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..utils.exceptions import DataLoadError
from ..utils.timing import timing_decorator
from config.config import CHUNK_SIZE, CHUNK_OVERLAP

@timing_decorator
def load_data(file_path: str | Path, max_lines: Optional[int] = None) -> pd.DataFrame:
    """
    Load and preprocess startup data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        max_lines: Maximum number of lines to load (optional)
        
    Returns:
        Preprocessed DataFrame containing startup descriptions
        
    Raises:
        DataLoadError: If there's an error loading or processing the data
    """
    try:
        df = pd.read_json(file_path)
        if max_lines:
            df = df.head(max_lines)
            
        df = df[['long_desc']]
        df.drop_duplicates(subset=['long_desc'], inplace=True)
        df = df[df['long_desc'].notna()]
        
        return df
    except ValueError as e:
        raise DataLoadError(f"Invalid JSON format: {e}")
    except FileNotFoundError as e:
        raise DataLoadError(f"File not found: {e}")
    except Exception as e:
        raise DataLoadError(f"Unexpected error loading data: {e}")

@timing_decorator
def create_documents(df: pd.DataFrame) -> list[Document]:
    """
    Create Document objects from DataFrame rows.
    
    Args:
        df: DataFrame containing startup descriptions
        
    Returns:
        List of Document objects
    """
    return [Document(page_content=row['long_desc']) for _, row in df.iterrows()]

@timing_decorator
def split_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents into smaller chunks for processing.
    
    Args:
        documents: List of Document objects
        
    Returns:
        List of split Document objects
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return text_splitter.split_documents(documents)
