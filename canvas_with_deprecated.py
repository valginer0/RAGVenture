
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import langchain
import langsmith
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import VectorStoreRetriever

# Load the dataset from Kaggle (assuming you've already downloaded it)
# data_path = "y_combinator_startups.csv"
data_path = "yc_startups.json"
df = pd.read_csv(data_path)

# Display the first few rows of the dataframe to understand the data
print("Loaded Data:")
print(df.head())

# Extracting relevant data columns for analysis
# Assuming columns such as 'Startup Name', 'Description', 'Funding', 'Sector', etc.
startup_data = df[['Startup Name', 'Description', 'Funding', 'Sector']]

# Setting up LangChain and LangSmith
# Initialize LangSmith for natural language processing
nlp = langsmith.LanguageProcessor()

# Creating embeddings for the data
embeddings = OpenAIEmbeddings()

# Convert startup descriptions into a list for embedding
documents = startup_data['Description'].tolist()

# Creating a vector store using FAISS
vector_store = FAISS.from_texts(documents, embeddings)

# Creating a retriever using VectorStoreRetriever
retriever = VectorStoreRetriever(vector_store=vector_store)

# Function to query startup information
def query_startup_info(query):
    """
    Queries the VectorStoreRetriever with a natural language query to find information about Y Combinator startups.
    """
    response = retriever.retrieve(query)
    return response

# Example usage
if __name__ == "__main__":
    user_query = "Which startups have received the most funding in the healthcare sector?"
    response = query_startup_info(user_query)
    print("\nResponse:")
    print(response)

# This code is a starting point and will need configuration depending on
# the exact requirements and data format of the Kaggle dataset.
# Ensure that all dependencies (LangChain, LangSmith, FAISS) are properly installed.

