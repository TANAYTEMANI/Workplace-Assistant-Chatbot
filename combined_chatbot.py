"""
Combined Chatbot - Workplace Assistant
Converted from Jupyter notebook to Python script
Uses GROQ API for LLM functionality
"""

import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")
DATA_PATH = "pdf_data"
CHROMA_PATH = "chroma"
JSON_PATH = "leavesEmployee.json"

# Initialize GROQ LLM
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name=GROQ_MODEL,
    temperature=0.7,
    max_tokens=512
)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def load_json_data():
    """Load employee data from JSON file."""
    try:
        with open(JSON_PATH, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: {JSON_PATH} not found")
        return None


def json_query(query_text, json_data):
    """
    Query employee data from JSON using GROQ.
    
    Args:
        query_text: The user's question
        json_data: The JSON data to query
        
    Returns:
        Response from the LLM
    """
    if not json_data:
        return "Error: No JSON data available"
    
    json_string = json.dumps(json_data, indent=2)
    
    prompt = f"""Here is the data: {json_string}. 
    Examine and process the input file to generate response. 
    Do not provide context to the input data. 
    Respond to the user's query based on the input data in the second person, addressing the user as 'you'. 
    Answer the question in 1 sentence. 
    Now, answer the following query: {query_text}."""
    
    response = llm.invoke(prompt)
    return response.content if hasattr(response, 'content') else str(response)


def load_pdf_documents():
    """Load PDF documents from the data directory."""
    try:
        document_loader = PyPDFDirectoryLoader(DATA_PATH)
        return document_loader.load()
    except Exception as e:
        print(f"Error loading PDFs: {e}")
        return []


def split_documents(documents):
    """Split documents into chunks."""
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def get_vector_store():
    """Get or create the Chroma vector store."""
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )


def policy_query(query_text):
    """
    Query policy documents using RAG with GROQ.
    
    Args:
        query_text: The user's question about policies
        
    Returns:
        Response from the LLM with sources
    """
    # Get vector store
    db = get_vector_store()
    
    # Retrieve relevant documents
    results = db.similarity_search_with_score(query_text, k=5)
    
    if not results:
        return "No relevant policy information found."
    
    # Prepare context from retrieved documents
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    # Create prompt template
    prompt_template = """Answer the question based only on the following context:
{context}

---

Answer the question based on the above context: {question}"""
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    formatted_prompt = prompt.format(context=context_text, question=query_text)
    
    # Get response from GROQ
    response = llm.invoke(formatted_prompt)
    response_text = response.content if hasattr(response, 'content') else str(response)
    
    # Extract sources
    sources = [doc.metadata.get("id", "Unknown") for doc, _score in results]
    
    formatted_response = f"Response: {response_text}\n\nSources: {sources}"
    return formatted_response


def main():
    """Main function to demonstrate chatbot capabilities."""
    print("=" * 60)
    print("Workplace Assistant Chatbot")
    print("Powered by GROQ API")
    print("=" * 60)
    
    # Test JSON Query
    print("\n### JSON Query Test ###")
    json_data = load_json_data()
    if json_data:
        test_query = "who is my manager?"
        print(f"Query: {test_query}")
        result = json_query(test_query, json_data)
        print(f"Response: {result}")
    
    # Test Policy Query
    print("\n### Policy Query Test ###")
    test_policy_query = "What is the marriage gift policy?"
    print(f"Query: {test_policy_query}")
    result = policy_query(test_policy_query)
    print(f"{result}")
    
    print("\n" + "=" * 60)
    print("Chatbot demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
