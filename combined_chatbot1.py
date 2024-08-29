import json
import argparse
import requests
from time import sleep
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.agents import Tool
from langchain.tools import BaseTool
from langchain.prompts import ChatPromptTemplate
from langchain.agents import initialize_agent
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

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
    
    #response = route_query(query_text, json_data)
    agent_executor = tool_app(query_text)
    response = agent_executor.invoke({"input": query_text})
    print(response)


# def route_query(query_text, json_data):
#     # Simple routing logic based on the query
#     if "policy" in query_text.lower() or "eligibility" in query_text.lower():
#         return doc_app(query_text)
#     else:
#         return json_app(json_data, query_text)
    

# def route_query(query_text, json_data):
#     # Simple routing logic based on the query
#     if "policy" in query_text.lower() or "eligibility" in query_text.lower():
#         return doc_app(query_text)
#     else:
#         return api_json_app(json_data, query_text)


def json_app(query_text):
    with open('leavesEmployee.json', 'r') as file:
        json_data = json.load(file)
    flat_json_string = json.dumps(json_data, indent=2)
    prompt = f"Here is the data: {flat_json_string}. Do not mention any other context other than the query response. Respond to the user's query in the second person, addressing the user as 'you'. Answer the question in 1 sentence without mentioning the data context. Now, answer the following query: {query_text}."
    
    llm = Ollama(model="llama2")  # Adjust if you have different models for JSON queries
    response = llm.generate(prompts=[prompt], max_tokens=50)
    generated_text = response.generations[0][0].text  # Adjust based on actual response structure
    return generated_text


def api_json_app(query_text):
    with open('leavesEmployee.json', 'r') as file:
        json_data = json.load(file)
    flat_json_string = json.dumps(json_data, indent=2)
    headers = {"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiMDFmNDRmODEtNTk3ZC00YjM4LWJiYTUtNmYxNGZjZWI3MjMxIiwidHlwZSI6ImFwaV90b2tlbiJ9.Bm3ALiGDdj-BzuhIy0QaxEPQMizBfUuigNuqE9FKOUU"}
    url = "https://api.edenai.run/v2/workflow/545a639b-6cd3-4858-bc30-58981d5ed0aa/execution/"
    payload = {'JsonInput': flat_json_string, 'QuestionInput': query_text}
    response = requests.post(url, json=payload, headers=headers)
    result = json.loads(response.text)
    sleep(5)
    execution_id = result['id']
    headers = {"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiMDFmNDRmODEtNTk3ZC00YjM4LWJiYTUtNmYxNGZjZWI3MjMxIiwidHlwZSI6ImFwaV90b2tlbiJ9.Bm3ALiGDdj-BzuhIy0QaxEPQMizBfUuigNuqE9FKOUU"}
    url = f"https://api.edenai.run/v2/workflow/545a639b-6cd3-4858-bc30-58981d5ed0aa/execution/{execution_id}/"
    response = requests.get(url, headers=headers)
    result = json.loads(response.text)
    generated_text = result['content']['results']['text__chat']['results'][0]['generated_text']
    return generated_text


def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings

def doc_app(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    model = Ollama(model="llama2")  # Adjust if you have different models for document queries
    response_text = model.invoke(prompt)
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    return formatted_response

def tool_app(query_text):
    json_tool = Tool(
        name="JsonTool",
        func=json_app,
        description="A LangChain tool designed to handle queries related to employee details. This tool takes JSON data as input, where the data includes various employee information such as names, positions, departments, and contact details. It uses this data to generate accurate and relevant responses to any queries concerning employee details, including remaining leave-related questions. Specifically, it can answer leave-related questions by checking the remaining leave fields and determining if an employee is eligible to take leave."
    )

    api_json_tool = Tool(
        name="APIJsonTool",
        func=api_json_app,
        description="A LangChain tool designed to handle queries related to employee details. This tool takes JSON data as input, where the data includes various employee information such as names, positions, departments, and contact details. It uses this data to generate accurate and relevant responses to any queries concerning employee details, including remaining leave-related questions. Specifically, it can answer leave-related questions by checking the remaining leave fields and determining if an employee is eligible to take leave."
    )


    doc_tool = Tool(
        name="DocumentsTool",
        func=doc_app,
        description="A LangChain tool designed to manage and respond to queries related to HR policies. This tool takes PDF files as input, where each file contains detailed information about various HR policies such as leaves description, insurance coverage, code of conduct, and more. It processes these PDF documents to generate precise and relevant responses to any questions concerning HR policies."
    )

    tools = [api_json_tool, doc_tool]
    #tools = [api_json_tool, doc_tool]

#     memory = ConversationBufferWindowMemory(
#         memory_key='chat_history',
#         k=3,
#         return_messages=True
#     )

#     conversational_agent = initialize_agent(
#         agent='chat-conversational-react-description',
#         tools=tools,
#         llm=Ollama(model="llama2"),
#         verbose=True,
#         max_iterations=3,
#         early_stopping_method='generate',
#         memory=memory
#     )
#     return conversational_agent

    llm=Ollama(model="llama2") 
    #prompt_template = hub.pull("hwchase17/react")


    template = '''Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Context: You are a highly knowledgeable assistant with access to two distinct types of information:
    1. **Employee Details**: This includes structured data about employees, such as their names, job titles, departments, and leave balances. The information is presented in a structured JSON format.
    2. **HR Policies**: This encompasses detailed policies and guidelines related to HR matters, including leave policies, insurance coverage, and workplace conduct, derived from various documents.
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the same input question that needs to be answered
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}'''

    prompt = PromptTemplate.from_template(template)
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    return agent_executor
    


if __name__ == "__main__":
    main()
