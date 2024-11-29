"""
To use langsmith set up the enviromnent variables :

LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="your lansmith api key"
LANGCHAIN_PROJECT="the name of your langsmith project"

"""

import logging
import time
from typing import List, Tuple, Optional, Any
import pandas as pd
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import json

from config.config import LOCAL_LANGUAGE_MODEL
from src.rag_startups.core.rag_chain import format_startup_idea, initialize_rag
from src.rag_startups.core.startup_metadata import StartupLookup
from src.rag_startups.data.loader import (
    create_documents,
    split_documents,
    StartupLookup,
)
from src.rag_startups.utils.spinner import Spinner
from src.rag_startups.utils.timing import timing_decorator

# Configure logging to suppress batch processing messages
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)


class CustomEmbeddingFunction:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts)

    def embed_query(self, text):
        return self.model.encode(text)


@timing_decorator
def create_and_split_document(df: pd.DataFrame):
    docs = create_documents(df)
    return split_documents(docs)


@timing_decorator
def embed(splits, model_name):
    try:
        embedding_model = SentenceTransformer(model_name)
        custom_embedder = CustomEmbeddingFunction(embedding_model)
        texts = [doc.page_content for doc in splits]
        return Chroma.from_texts(texts, embedding=custom_embedder)
    except Exception as e:
        logging.error(f"Error during embedding: {e}")
        return None


@timing_decorator
def setup_retriever(vectorstore):
    try:
        return vectorstore.as_retriever()
    except Exception as e:
        logging.error(f"Error setting up retriever: {e}")
        return None


@timing_decorator
def initialize_embeddings(df: pd.DataFrame, model_name: str = "all-MiniLM-L6-v2"):
    """Initialize embeddings and retriever. This should be called once at startup."""
    splits = create_and_split_document(df)
    if not splits:
        raise ValueError("Failed to split documents.")

    print("\nInitializing embeddings for faster startup idea generation...")
    with Spinner("Computing embeddings and setting up retriever"):
        vectorstore = embed(splits, model_name)
        if vectorstore is None:
            raise ValueError("Failed to create vectorstore.")

        retriever = setup_retriever(vectorstore)
        if retriever is None:
            raise ValueError("Failed to set up retriever.")
    print("Initialization complete!\n")

    return retriever


def get_prompt_content(prompt, question, context_docs):
    formatted_context = "\n\n".join(doc.page_content for doc in context_docs)
    prompt_input = prompt.format_messages(question=question, context=formatted_context)
    return prompt_input[0].content


@timing_decorator
def rag_chain_local(question, generator, prompt, retriever, lookup=None, num_ideas=3):
    try:
        context_docs = retriever.invoke(question)
        formatted_ideas = []
        seen_companies = set()

        for doc in context_docs:
            # Format the idea using the correct function
            sections = format_startup_idea(doc.page_content, retriever, lookup)

            # Skip if we've already seen this company
            if sections["Company"] in seen_companies:
                continue

            seen_companies.add(sections["Company"])

            # Format the idea with proper sections
            formatted_idea = f"\n{'='*50}\nStartup Idea #{len(formatted_ideas)+1}:\n"
            formatted_idea += f"Company: {sections['Company']}\n\n"
            formatted_idea += f"PROBLEM/OPPORTUNITY:\n{sections['Problem']}\n\n"
            formatted_idea += f"SOLUTION:\n{sections['Solution']}\n\n"
            formatted_idea += f"TARGET MARKET:\n{sections['Market']}\n\n"
            formatted_idea += f"UNIQUE VALUE:\n{sections['Value']}"

            formatted_ideas.append(formatted_idea)

            # Stop after num_ideas unique companies
            if len(formatted_ideas) >= num_ideas:
                break

        return (
            "Here are the most relevant startup ideas from YC companies:\n"
            + "\n".join(formatted_ideas)
        )
    except Exception as e:
        logging.error(f"Error in RAG chain: {e}")
        return ""


@timing_decorator
def calculate_result(
    question: str,
    retriever: Any,
    json_data: list,
    prompt_messages: List[Tuple[str, str]],
    model_name: str = "all-MiniLM-L6-v2",
    lookup: Optional[StartupLookup] = None,
    num_ideas: int = 3,
) -> str:
    """Calculate result using the RAG pipeline."""
    # Initialize lookup if not provided
    if lookup is None:
        lookup = StartupLookup(json_data)

    generator = pipeline(
        "text-generation", model=LOCAL_LANGUAGE_MODEL, pad_token_id=50256
    )
    prompt = ChatPromptTemplate.from_messages(prompt_messages)
    result = rag_chain_local(
        question, generator, prompt, retriever, lookup=lookup, num_ideas=num_ideas
    )
    return result
