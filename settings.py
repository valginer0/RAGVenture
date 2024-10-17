# settings.py

file_path = './yc_startups.json'
model_name = 'all-MiniLM-L6-v2'
local_language_model_name = 'gpt2'

limitation_of_number_of_lines = 5000000

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

question = "Generate a company idea for the Real Estate based on provided context"
# question = "Generate a company idea for the Automobile industry based on provided context"
# question = "Generate a company idea for the HR industry based on provided context"
