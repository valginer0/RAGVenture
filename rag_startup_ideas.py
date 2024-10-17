"""
To use langsmith set up the enviromnent variables :

LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="your lansmith api key"
LANGCHAIN_PROJECT="the name of your langsmith project"

"""

import os
import pandas as pd
import time
import logging
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.docstore.document import Document

from langsmith import traceable

# my_longsmith_api_key = os.environ['LANGCHAIN_API_KEY']

from functools import wraps
from settings import file_path, model_name, local_language_model_name, prompt_messages, question, \
    limitation_of_number_of_lines

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
def load_data(file_path: str):
    df = pd.read_json(file_path).head(limitation_of_number_of_lines)
    df = df[['long_desc']]
    df.drop_duplicates(subset=['long_desc'], inplace=True)
    df = df[df['long_desc'].notna()]
    return df


@timing_decorator
def create_and_split_document(df: pd.DataFrame):
    docs = [Document(page_content=row['long_desc']) for _, row in df.iterrows()]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(docs)


@timing_decorator
def embed(splits, model_name):
    embedding_model = SentenceTransformer(model_name)
    custom_embedder = CustomEmbeddingFunction(embedding_model)
    texts = [doc.page_content for doc in splits]
    return Chroma.from_texts(texts, embedding=custom_embedder)


@timing_decorator
def setup_retriever(vectorstore):
    return vectorstore.as_retriever()


def get_prompt_content(prompt, question, context_docs):
    formatted_context = "\n\n".join(doc.page_content for doc in context_docs)
    prompt_input = prompt.format_messages(question=question, context=formatted_context)
    return prompt_input[0].content


@traceable
@timing_decorator
def rag_chain_local(question, generator, prompt, retriever):
    # context_docs = retriever.get_relevant_documents(question)
    context_docs = retriever.invoke(question)
    prompt_content = get_prompt_content(prompt, question, context_docs)
    return generator(prompt_content, max_new_tokens=50, num_return_sequences=1, pad_token_id=50256, truncation=True)[0][
        'generated_text']


df = load_data(file_path)

splits = create_and_split_document(df)
vectorstore = embed(splits, model_name)

retriever = setup_retriever(vectorstore)

generator = pipeline("text-generation", model=local_language_model_name, pad_token_id=50256)

prompt = ChatPromptTemplate.from_messages(prompt_messages)

result = rag_chain_local(question, generator, prompt, retriever)
print(result)
