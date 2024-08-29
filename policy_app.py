import argparse
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings


# from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def get_embedding_function():
    embeddings = OllamaEmbeddings(model = "nomic-embed-text")
    return embeddings

def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = Ollama(model="llama2")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()




# #---------------------------------------------------------------------------------



# import requests

# def get_embedding_function():
#     # Placeholder function to get the embedding function
#     pass

# class Chroma:
#     def __init__(self, persist_directory, embedding_function):
#         # Placeholder class for Chroma
#         pass
    
#     def similarity_search_with_score(self, query_text, k):
#         # Placeholder method to perform similarity search
#         pass

# class ChatPromptTemplate:
#     @staticmethod
#     def from_template(template):
#         # Placeholder method to create a prompt template
#         return ChatPromptTemplate()
    
#     def format(self, context, question):
#         # Placeholder method to format the prompt
#         return f"{context}\n\n{question}"

# CHROMA_PATH = "path_to_chroma_db"
# PROMPT_TEMPLATE = "Your prompt template"

# def query_rag(query_text: str):
#     # Prepare the DB.
#     embedding_function = get_embedding_function()
#     db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

#     results = db.similarity_search_with_score(query_text, k=5)

#     context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
#     prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#     prompt = prompt_template.format(context=context_text, question=query_text)

#     # Use Hugging Face Inference API
#     api_url = "https://api-inference.huggingface.co/models/your-model-name"
#     api_token = "your_huggingface_api_token"

#     headers = {
#         "Authorization": f"Bearer {api_token}",
#         "Content-Type": "application/json"
#     }

#     data = {
#         "inputs": prompt,
#         "options": {"wait_for_model": True}
#     }

#     response = requests.post(api_url, headers=headers, json=data)
#     response_text = response.json()["generated_text"]

#     sources = [doc.metadata.get("id", None) for doc, _score in results]
#     formatted_response = f"Response: {response_text}\nSources: {sources}"
#     print(formatted_response)
#     return response_text

# # Example usage
# # response = query_rag("Your query here")
# # print(response)
