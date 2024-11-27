"""
To use langsmith set up the enviromnent variables :

LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="your lansmith api key"
LANGCHAIN_PROJECT="the name of your langsmith project"

"""

import logging
import time
from functools import wraps

import pandas as pd
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from settings import local_language_model_name, limitation_of_number_of_lines

logging.basicConfig(level=logging.INFO)


class CustomEmbeddingFunction:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return [self.model.encode(text) for text in texts]

    def embed_query(self, text):
        return self.model.encode(text)


def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result

    return wrapper


@timing_decorator
def load_data(file_path: str, limitation_of_number_of_lines):
    try:
        df = pd.read_json(file_path).head(limitation_of_number_of_lines)
        df = df[['long_desc']]
        df.drop_duplicates(subset=['long_desc'], inplace=True)
        df = df[df['long_desc'].notna()]
        return df
    except ValueError as e:
        logging.error(f"Error loading data: The file might not be in valid JSON format. Details: {e}")
    except FileNotFoundError as e:
        logging.error(f"Error loading data: File not found. Details: {e}")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
    return pd.DataFrame()


@timing_decorator
def create_and_split_document(df: pd.DataFrame):
    docs = [Document(page_content=row['long_desc']) for _, row in df.iterrows()]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(docs)


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


def get_prompt_content(prompt, question, context_docs):
    formatted_context = "\n\n".join(doc.page_content for doc in context_docs)
    prompt_input = prompt.format_messages(question=question, context=formatted_context)
    return prompt_input[0].content


@traceable
@timing_decorator
def rag_chain_local(question, generator, prompt, retriever):
    try:
        context_docs = retriever.invoke(question)
        prompt_content = get_prompt_content(prompt, question, context_docs)
        return \
        generator(prompt_content, max_new_tokens=50, num_return_sequences=1, pad_token_id=50256, truncation=True)[0][
            'generated_text']
    except Exception as e:
        logging.error(f"Error in RAG chain: {e}")
        return ""


def calculate_result(question, file_path, prompt_messages, model_name='all-MiniLM-L6-v2', limitation_of_number_of_lines=500_000):
    df = load_data(file_path, limitation_of_number_of_lines)
    if df.empty:
        return "Error: Failed to load data."

    splits = create_and_split_document(df)
    if not splits:
        return "Error: Failed to split documents."

    vectorstore = embed(splits, model_name)
    if vectorstore is None:
        return "Error: Failed to create vectorstore."

    retriever = setup_retriever(vectorstore)
    if retriever is None:
        return "Error: Failed to set up retriever."

    generator = pipeline("text-generation", model=local_language_model_name, pad_token_id=50256)
    prompt = ChatPromptTemplate.from_messages(prompt_messages)
    result = rag_chain_local(question, generator, prompt, retriever)
    return result
