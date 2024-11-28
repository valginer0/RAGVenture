"""Tests for RAG chain functionality."""
import sys
from pathlib import Path
import pandas as pd
import pytest
from langchain.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate
from transformers import pipeline

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from embed_master import create_and_split_document, embed, setup_retriever, rag_chain_local
from src.rag_startups.core.startup_metadata import StartupLookup
from settings import local_language_model_name

def test_create_and_split_document():
    """Test document creation and splitting."""
    # Create a sample dataframe
    df = pd.DataFrame({
        'long_desc': [
            'This is a test description for company A',
            'This is another test description for company B'
        ]
    })
    
    # Test splitting
    splits = create_and_split_document(df)
    
    # Basic assertions
    assert splits is not None
    assert len(splits) >= 2  # Should have at least our 2 test documents
    assert all(hasattr(doc, 'page_content') for doc in splits)
    
    # Content assertions
    contents = [doc.page_content for doc in splits]
    assert any('company A' in content for content in contents)
    assert any('company B' in content for content in contents)

def test_embed():
    """Test document embedding."""
    # Create test documents
    docs = [
        Document(page_content="Test document A about AI technology"),
        Document(page_content="Test document B about blockchain")
    ]
    
    # Test embedding
    vectorstore = embed(docs, model_name='all-MiniLM-L6-v2')
    
    # Basic assertions
    assert vectorstore is not None
    
    # Test similarity search functionality
    results = vectorstore.similarity_search("AI", k=1)
    assert len(results) == 1
    assert "AI" in results[0].page_content

def test_setup_retriever():
    """Test retriever setup."""
    # First create a vectorstore with test documents
    docs = [
        Document(page_content="Test document A about AI technology"),
        Document(page_content="Test document B about blockchain")
    ]
    vectorstore = embed(docs, model_name='all-MiniLM-L6-v2')
    
    # Test retriever setup
    retriever = setup_retriever(vectorstore)
    
    # Basic assertions
    assert retriever is not None
    
    # Test retrieval functionality
    retrieved_docs = retriever.invoke("AI technology")
    assert len(retrieved_docs) > 0
    assert any("AI" in doc.page_content for doc in retrieved_docs)

def test_rag_chain_local():
    """Test RAG chain end-to-end."""
    # Create test documents and vectorstore
    docs = [
        Document(page_content="Company A builds AI solutions for healthcare, focusing on medical imaging."),
        Document(page_content="Company B develops blockchain technology for supply chain tracking.")
    ]
    vectorstore = embed(docs, model_name='all-MiniLM-L6-v2')
    retriever = setup_retriever(vectorstore)
    
    # Set up generator and prompt
    generator = pipeline("text-generation", model=local_language_model_name, pad_token_id=50256)
    prompt_messages = [
        ("system", """
            You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the question.
            If you don't know the answer, just say that you don't know.
            Use three sentences maximum and keep the answer concise.
            Question: {question}
            Context: {context}
            Answer:""")
    ]
    prompt = ChatPromptTemplate.from_messages(prompt_messages)
    
    # Test RAG chain
    result = rag_chain_local("Tell me about AI companies", generator, prompt, retriever)
    
    # Basic assertions
    assert result is not None
    assert isinstance(result, str)
    assert len(result.strip()) > 0  # Should not be empty

@pytest.fixture
def sample_startup_data():
    """Sample startup data for testing."""
    return [
        {
            'name': 'AI Company',
            'description': 'An AI company that does machine learning.',
            'long_desc': 'An AI company that does machine learning. Using cutting-edge algorithms to solve complex problems. Targeting enterprise customers.',
            'industry': 'AI'
        },
        {
            'name': 'Blockchain Startup',
            'description': 'A blockchain company for secure transactions.',
            'long_desc': 'A blockchain company for secure transactions. Providing decentralized solutions for financial services. Focused on institutional clients.',
            'industry': 'Blockchain'
        }
    ]

@pytest.fixture
def sample_startup_lookup(sample_startup_data):
    """Sample StartupLookup instance for testing."""
    return StartupLookup(sample_startup_data)

def test_rag_chain_with_lookup(sample_startup_lookup):
    """Test RAG chain with startup lookup integration."""
    # Create test documents with long_desc
    docs = [
        Document(page_content="An AI company that does machine learning. Using cutting-edge algorithms to solve complex problems. Targeting enterprise customers."),
        Document(page_content="A blockchain company for secure transactions. Providing decentralized solutions for financial services. Focused on institutional clients.")
    ]
    
    # Create vectorstore and retriever
    vectorstore = embed(docs, model_name='all-MiniLM-L6-v2')
    retriever = setup_retriever(vectorstore)
    
    # Create generator and prompt
    generator = pipeline("text-generation", model=local_language_model_name, pad_token_id=50256)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Generate a startup idea based on: {context}")
    ])
    
    # Test RAG chain with lookup
    result = rag_chain_local(
        "Generate an AI startup idea",
        generator,
        prompt,
        retriever,
        lookup=sample_startup_lookup
    )
    
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0

def test_format_startup_idea_with_lookup(sample_startup_lookup):
    """Test formatting startup ideas with lookup."""
    from src.rag_startups.core.rag_chain import format_startup_idea
    
    # Create test documents and retriever with long_desc
    docs = [Document(page_content="An AI company that does machine learning. Using cutting-edge algorithms to solve complex problems. Targeting enterprise customers.")]
    vectorstore = embed(docs, model_name='all-MiniLM-L6-v2')
    retriever = setup_retriever(vectorstore)
    
    # Test formatting with lookup
    result = format_startup_idea(
        "An AI company that does machine learning. Using cutting-edge algorithms to solve complex problems. Targeting enterprise customers.",
        retriever,
        startup_lookup=sample_startup_lookup
    )
    
    assert result is not None
    assert isinstance(result, dict)
    assert 'Company' in result
    assert 'Problem' in result
    assert 'Solution' in result
    assert 'Market' in result
    assert 'Value' in result
    assert result['Company'] == 'AI Company'  # Should match the lookup data
