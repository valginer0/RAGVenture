"""Data loading and processing functionality."""
from pathlib import Path
from typing import Optional, Tuple, Union, List
import pandas as pd
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..core.startup_metadata import StartupLookup
from ..utils.exceptions import DataLoadError
from ..utils.timing import timing_decorator
from config.config import CHUNK_SIZE, CHUNK_OVERLAP

@timing_decorator
def load_data(file_path: str | Path, max_lines: Optional[int] = None) -> Tuple[pd.DataFrame, list]:
    """
    Load and preprocess startup data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        max_lines: Maximum number of lines to load (optional)
        
    Returns:
        Tuple of (preprocessed DataFrame, raw JSON data)
        
    Raises:
        DataLoadError: If there's an error loading or processing the data
    """
    try:
        # Load raw JSON first
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = pd.read_json(f)
            json_data = raw_data.to_dict('records')
        
        # Create DataFrame for RAG
        df = raw_data.copy()
        if max_lines:
            df = df.head(max_lines)
            json_data = json_data[:max_lines]
            
        # Keep only necessary columns for RAG
        if 'long_desc' in df.columns:
            df = df[['long_desc']]
        elif 'description' in df.columns:
            df = df[['description']]
        else:
            raise DataLoadError("No description column found in data")
            
        df.drop_duplicates(subset=df.columns[0], inplace=True)
        df = df[df[df.columns[0]].notna()]
        
        return df, json_data
    except ValueError as e:
        raise DataLoadError(f"Invalid JSON format: {e}")
    except FileNotFoundError as e:
        raise DataLoadError(f"File not found: {e}")
    except Exception as e:
        raise DataLoadError(f"Unexpected error loading data: {e}")

@timing_decorator
def create_documents(texts: Union[pd.DataFrame, List[str]]) -> list[Document]:
    """
    Create Document objects from texts.

    Args:
        texts: Either a DataFrame containing startup descriptions or a list of text strings

    Returns:
        List of Document objects
    """
    if isinstance(texts, pd.DataFrame):
        return [Document(page_content=row[texts.columns[0]]) for _, row in texts.iterrows()]
    else:
        return [Document(page_content=text) for text in texts if text]

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

@timing_decorator
def initialize_startup_lookup(json_data: list) -> StartupLookup:
    """
    Initialize the startup lookup from JSON data.
    
    Args:
        json_data: List of startup JSON records
        
    Returns:
        Populated StartupLookup instance
    """
    lookup = StartupLookup()
    for item in json_data:
        desc = item.get('long_desc', item.get('description', ''))
        if desc:
            lookup.add_startup(desc, item)
    return lookup