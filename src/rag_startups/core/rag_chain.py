"""Core RAG chain functionality."""
from typing import Any, List, Optional, Tuple
import re

from langchain_core.prompts import ChatPromptTemplate
from transformers import pipeline

from ..data.loader import load_data, create_documents, split_documents
from ..embeddings.embedding import create_vectorstore, setup_retriever
from ..utils.exceptions import ModelError
from ..utils.timing import timing_decorator
from config.config import (
    DEFAULT_PROMPT_TEMPLATE,
    LOCAL_LANGUAGE_MODEL,
    MAX_LINES
)

# Common words that should not be considered company names
COMMON_WORDS = {
    # Articles and basic words
    'the', 'a', 'an', 'and', 'or', 'but', 'if', 'in', 'on', 'at', 'by', 'to', 'for',
    # Pronouns
    'we', 'i', 'you', 'they', 'it', 'he', 'she',
    # Common verbs
    'help', 'provide', 'get', 'use', 'make', 'build', 'create', 'start', 'learn', 'work',
    'exists', 'is', 'are', 'was', 'be',
    # Product/service words
    'platform', 'solution', 'service', 'product', 'company', 'business', 'startup',
    'software', 'desk', 'system', 'tool', 'app', 'application',
    # Other common words
    'that', 'this', 'what', 'which', 'new', 'first', 'best'
}

def extract_company_name(text):
    """Extract company name from the first sentence, with better filtering."""
    if not text:
        return ""
    
    # Split into words and look for first word that could be a company name
    words = text.split()
    
    # Check first word specially - most company names appear first
    if words and len(words[0]) > 1:
        clean_word = words[0].strip('.,!?:;"\'').strip()
        if (clean_word and 
            clean_word.lower() not in COMMON_WORDS and
            not clean_word.lower().startswith('help') and  # Special case for "help desk" etc.
            clean_word[0].isupper()):  # Company names start with uppercase
            return clean_word
    
    # If first word wasn't a company name, check second word only if first word is "the"
    if len(words) > 1 and words[0].lower() == 'the':
        clean_word = words[1].strip('.,!?:;"\'').strip()
        if (clean_word and 
            clean_word.lower() not in COMMON_WORDS and
            not clean_word.lower().startswith('help') and
            clean_word[0].isupper()):
            return clean_word
            
    return ""

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

def format_startup_idea(content: str) -> dict:
    """Format a startup description into a structured format."""
    # Clean and join lines, removing empty ones
    lines = [line.strip() for line in content.split("\n") if line.strip()]
    content = " ".join(lines)
    
    # Try to identify the main components
    parts = content.split(".")
    description = parts[0].strip()
    
    # Extract company name using the improved function
    company_name = extract_company_name(description)
    if not company_name and len(parts) > 1:
        # Try extracting from second part if first part failed
        company_name = extract_company_name(parts[1])
    
    # Get main description by removing company name
    main_desc = description
    if company_name:
        # Remove company name while preserving the rest of the sentence
        main_desc = description.replace(company_name, "", 1).strip()
        if main_desc.startswith('exists to '):
            main_desc = f"{company_name} {main_desc}"
    
    # Fix descriptions that start with verbs by adding subject
    if main_desc.lower().startswith(('help', 'provide', 'create', 'build', 'make', 'is')):
        main_desc = f"{company_name} {main_desc}" if company_name else main_desc
    
    # Remove "Link exists" and similar phrases from start
    main_desc = re.sub(r'^(link exists|exists)\s+to\s+', '', main_desc, flags=re.IGNORECASE)
        
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
    main_desc = main_desc.rstrip(".") + "."
    solution_details = solution_details.rstrip(".") + "." if solution_details else f"Provides innovative solutions for {target_market.lower()}"
    target_market = target_market.rstrip(".") + "."
    
    return {
        'Company': company_name,
        'Problem': main_desc,
        'Solution': solution_details,
        'Market': target_market,
        'Value': value_details
    }

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
        context_docs = retriever.get_relevant_documents(question)
        
        # Format the startup ideas
        formatted_ideas = []
        seen_companies = set()  # Track companies we've already seen
        
        for doc in context_docs:
            # Format the idea
            sections = format_startup_idea(doc.page_content)
            
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

def looks_like_company_name(word: str) -> bool:
    """Check if a word looks like a company name."""
    # Remove punctuation and convert to lower
    word = word.lower().strip('.,!?')
    
    # Common patterns that indicate it's NOT a company name
    if (word in COMMON_WORDS or
        word.startswith(('help', 'provide', 'create', 'build', 'make', 'is', 'are', 'was', 'were')) or
        len(word) < 2):  # Too short to be a company name
        return False
            
    return True
