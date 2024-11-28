"""Core RAG chain functionality."""
from typing import Any, List, Optional, Tuple
import re

from langchain_core.prompts import ChatPromptTemplate
from transformers import pipeline

from ..data.loader import load_data, create_documents, split_documents, initialize_startup_lookup
from ..embeddings.embedding import create_vectorstore, setup_retriever
from ..utils.exceptions import ModelError
from ..utils.timing import timing_decorator
from config.config import (
    DEFAULT_PROMPT_TEMPLATE,
    LOCAL_LANGUAGE_MODEL,
    MAX_LINES
)

# Global lookup instance
startup_lookup = None

@timing_decorator
def initialize_rag(file_path: str, max_lines: Optional[int] = None):
    """Initialize RAG system and lookup."""
    global startup_lookup
    
    # Load data
    df, json_data = load_data(file_path, max_lines)
    
    # Initialize lookup
    startup_lookup = initialize_startup_lookup(json_data)
    
    # Create and prepare documents
    documents = create_documents(df)
    splits = split_documents(documents)
    
    # Setup retriever
    vectorstore = create_vectorstore(splits)
    return setup_retriever(vectorstore)

def get_similar_description(description: str, retriever: Any) -> Optional[str]:
    """Get most similar description from the vector store."""
    if not description:
        return None
        
    similar_docs = retriever.invoke(description)
    return similar_docs[0].page_content if similar_docs else None

def format_startup_idea(description: str, retriever: Any, startup_lookup: Optional[Any] = None) -> dict:
    """Format a startup description into a structured format."""
    # Get similar description from RAG
    similar_desc = get_similar_description(description, retriever)
    
    # Get metadata if we found a similar description and have a lookup
    metadata = startup_lookup.get_metadata(similar_desc) if (similar_desc and startup_lookup) else None
    
    # Use actual company name from metadata if available
    company_name = metadata.name if metadata else ""
    
    # Clean and join lines, removing empty ones
    lines = [line.strip() for line in description.split("\n") if line.strip()]
    content = " ".join(lines)
    
    # Try to identify the main components
    parts = content.split(".")
    main_desc = parts[0].strip()
    
    # Get main description
    main_desc = main_desc.rstrip(".") + "."
    
    # Analyze the remaining parts for additional details
    remaining_text = ". ".join(parts[1:]).strip()
    
    # Try to identify solution details (often follows the main description)
    solution_details = ""
    market_details = ""
    value_details = ""
    
    if remaining_text:
        remaining_parts = remaining_text.split(".")
        if len(remaining_parts) > 0:
            solution_details = remaining_parts[0].strip()
        if len(remaining_parts) > 1:
            market_details = remaining_parts[1].strip()
        if len(remaining_parts) > 2:
            value_details = ". ".join(remaining_parts[2:]).strip()
    
    # Try to extract target market from description
    target_market = "Organizations and stakeholders in this space"
    if "for" in main_desc.lower():
        extracted_market = main_desc.lower().split("for", 1)[1].strip()
        if len(extracted_market) > 3:  # Only use if meaningful
            target_market = extracted_market
    elif market_details:
        target_market = market_details
    
    # Format value details
    value_details = format_value_details(value_details)
    if not value_details:
        value_details = solution_details if solution_details else main_desc
    
    # Ensure proper sentence endings
    solution_details = solution_details.rstrip(".") + "." if solution_details else f"Provides innovative solutions for {target_market.lower()}"
    target_market = target_market.rstrip(".") + "."
    
    return {
        'Company': company_name,
        'Problem': main_desc,
        'Solution': solution_details,
        'Market': target_market,
        'Value': value_details
    }

def format_value_details(value_details):
    """Format value details, handling quotes and bullet points."""
    if not value_details:
        return ""
        
    # Remove quotes at start and end while preserving internal quotes
    value_details = value_details.strip()
    while value_details.startswith('"'):
        value_details = value_details[1:].lstrip()
    while value_details.endswith('"'):
        value_details = value_details[:-1].rstrip()
        
    # Format bullet points if multiple sentences
    sentences = [s.strip() for s in value_details.split(".") if s.strip()]
    if len(sentences) > 1:
        formatted_sentences = []
        for sentence in sentences:
            if sentence:  # Skip empty sentences
                # Capitalize first letter if it's not already
                if sentence[0].islower():
                    sentence = sentence[0].upper() + sentence[1:]
                formatted_sentences.append(f"• {sentence}")
        value_details = "\n".join(formatted_sentences)
    
    return value_details

@timing_decorator
def rag_chain_local(
    question: str,
    generator: Any,
    prompt_template: str,
    retriever: Any,
    max_lines: int = MAX_LINES,
) -> str:
    """Execute the RAG chain locally."""
    try:
        # Get relevant documents
        context_docs = retriever.invoke(question)
        
        # Format the startup ideas
        formatted_ideas = []
        seen_companies = set()  # Track companies we've already seen
        
        for doc in context_docs:
            # Format the idea
            sections = format_startup_idea(doc.page_content, retriever)
            
            # Skip if we've already seen this company
            if sections['Company'] in seen_companies:
                continue
                
            seen_companies.add(sections['Company'])
            
            # Format the idea with proper sections
            formatted_idea = f"\n{'='*50}\nStartup Idea #{len(formatted_ideas)+1}:\n"
            formatted_idea += f"Company: {sections['Company']}\n\n"
            formatted_idea += f"PROBLEM/OPPORTUNITY:\n{sections['Problem']}\n\n"
            formatted_idea += f"SOLUTION:\n{sections['Solution']}\n\n"
            formatted_idea += f"TARGET MARKET:\n{sections['Market']}\n\n"
            formatted_idea += f"UNIQUE VALUE:\n{sections['Value']}"
            
            formatted_ideas.append(formatted_idea)
            
            # Stop after 3 unique companies
            if len(formatted_ideas) >= 3:
                break
        
        return "Here are the most relevant startup ideas from YC companies:\n" + "\n".join(formatted_ideas)
        
    except Exception as e:
        raise ModelError(f"Error in RAG chain execution: {str(e)}")

@timing_decorator
def calculate_result(
    question: str,
    file_path: str,
    prompt_messages: List[Tuple[str, str]],
    model_name: str = LOCAL_LANGUAGE_MODEL,
    max_lines: Optional[int] = MAX_LINES
) -> str:
    """
    Calculate result using the RAG pipeline.
    
    Args:
        question: User's question
        file_path: Path to the data file
        prompt_messages: List of prompt message tuples
        model_name: Name of the language model
        max_lines: Maximum number of lines to process
        
    Returns:
        Generated answer
        
    Raises:
        Various exceptions from component functions
    """
    # Load and process data
    df = load_data(file_path, max_lines)
    docs = create_documents(df)
    splits = split_documents(docs)
    
    # Create vector store and retriever
    texts = [doc.page_content for doc in splits]
    vectorstore = create_vectorstore(texts)
    retriever = setup_retriever(vectorstore)
    
    # Set up language model
    try:
        generator = pipeline(
            'text-generation',
            model=model_name
        )
    except Exception as e:
        raise ModelError(f"Failed to load language model: {e}")
    
    # Get prompt template from messages
    prompt_template = prompt_messages[0][1] if prompt_messages else DEFAULT_PROMPT_TEMPLATE
    
    # Execute RAG chain
    return rag_chain_local(question, generator, prompt_template, retriever)