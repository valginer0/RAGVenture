"""
To use langsmith set up the enviromnent variables :

LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="your lansmith api key"
LANGCHAIN_PROJECT="the name of your langsmith project"

"""

from embed_master import calculate_result

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Please provide an industry. Example: python rag_startup_ideas.py AI")
        sys.exit(1)
        
    industry = sys.argv[1]
    question = f"Generate a company idea for the {industry} industry based on provided context"
    file_path = './yc_startups.json'
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

    result = calculate_result(question, file_path, prompt_messages)
    print(result)
