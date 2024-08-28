import streamlit as st
from langchain.agents import initialize_agent, AgentType
from langchain import OpenAI
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.llms import HuggingFaceEndpoint
import csv
import os
import requests
from bs4 import BeautifulSoup
import time
from dotenv import load_dotenv

load_dotenv()

# Set environment variables
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = 'playground'

# Set up API keys and repository ID
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv('HUGGINGFACEHUB_API_TOKEN')
repo_id = "mistralai/Mistral-Nemo-Instruct-2407"

# Initialize the LLM with Hugging Face Endpoint
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_length=200,
    temperature=0.1,
    token=os.getenv('HUGGINGFACEHUB_API_TOKEN'),
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
)

# Initialize DuckDuckGo Search API Wrapper
wrapper = DuckDuckGoSearchAPIWrapper(region="de-de", time="d", max_results=2)

def search_duckduckgo(query: str) -> str:
    results = wrapper.run(query)
    if not results:
        return "No results found."
    return results

duckduckgo_tool = Tool(
    name="DuckDuckGo Search",
    func=search_duckduckgo,
    description="Use this tool to search the web using DuckDuckGo."
)

# Tool 1: Calculator
def calculate_expression(expression: str) -> str:
    try:
        result = eval(expression)
        return f"The result of the expression '{expression}' is {result}."
    except Exception as e:
        return f"Error in calculation: {str(e)}"

math_tool = Tool(
    name="Calculator",
    func=calculate_expression,
    description="Use this tool to perform mathematical calculations."
)

# Tool 2: Web Search
def web_search(query: str) -> str:
    search_url = f"https://www.google.com/search?q={query}&tbs=qdr:y"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)

    soup = BeautifulSoup(response.text, "html.parser")
    results = soup.find_all('h3')

    if not results:
        return "No results found."

    return f"Top result for '{query}': {results[0].text}"

search_tool = Tool(
    name="Web Search",
    func=web_search,
    description="Use this tool to search the web using Google."
)

# Initialize the agent with tools
tools = [
    math_tool, duckduckgo_tool, search_tool
]

# Setup memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize the agent with memory and tools
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory
)

# Save input and output to CSV with headers and prediction time
def save_to_csv(user_input: str, agent_response: str, prediction_time: float):
    file_exists = os.path.isfile('agent_interaction_log.csv')
    with open('agent_interaction_log.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Write the header if the file does not exist
            writer.writerow(["User Input", "Agent Response", "Prediction Time (seconds)", "Timestamp"])
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        writer.writerow([user_input, agent_response, round(prediction_time, 2), current_time])

# Streamlit App
st.title("Advanced Chatbot ")
st.write("This is a chatbot powered by Rohan.")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Streamed response emulator
def response_generator(response_text):
    for word in response_text.split():
        yield word + " "
        time.sleep(0.05)

# Accept user input at the bottom
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Start timer for prediction time
    start_time = time.time()

    # Run the agent with the input
    agent_response = agent.run(input=prompt)

    # Calculate prediction time
    prediction_time = time.time() - start_time

    # Display assistant response in chat message container with streaming
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_text = ""
        for word in response_generator(agent_response):
            response_text += word
            response_placeholder.markdown(response_text)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    
    # Save to CSV with prediction time
    save_to_csv(prompt, response_text, prediction_time)
