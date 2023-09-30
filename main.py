from langchain.agents import Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain import hub
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.utilities import SerpAPIWrapper, SQLDatabase, WikipediaAPIWrapper
from langchain_experimental.sql import SQLDatabaseChain
from langchain.tools import WikipediaQueryRun
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.tools.render import format_tool_to_openai_function
import os
import langchain
import openai 
import streamlit as st

# Set your OpenAI API Key
os.environ["OPENAI_API_KEY"] = "Your_OpenAI_API_Key"

# Initialize the ChatOpenAI model
llm = ChatOpenAI(temperature=0, model='gpt-4-0613')

# Initialize external tools
search = SerpAPIWrapper(serpapi_api_key="Your_SerpAPI_Key")
db = SQLDatabase.from_uri("sqlite:///chinook.db")
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# Define tools
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful for answering questions about real-time events. Ask targeted questions."
    ),
    Tool(
        name="Wikipedia",
        func=wikipedia.run,
        description="Useful for answering questions about the big picture or background of something."
    ),
    Tool(
        name="MusicSales",
        func=db_chain.run,
        description="Useful for answering questions about music sales in a store. Follows strict table info."
    )
]

# Define a chat prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a useful assistant."),
    MessagesPlaceholder(variable_name='chat_history'),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Bind tools to the language model
llm_with_tools = llm.bind(
    functions=[format_tool_to_openai_function(t) for t in tools]
)

# Define the agent pipeline
agent = {
    "input": lambda x: x["input"],
    "agent_scratchpad": lambda x: format_to_openai_functions(x['intermediate_steps']),
    "chat_history": lambda x: x["chat_history"]
} | prompt | llm_with_tools | OpenAIFunctionsAgentOutputParser()

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)

# Streamlit setup
st.title("💬 Chatbot") 

# Initialize or retrieve messages from session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Display chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Initialize or retrieve the agent executor from session state
if "agent_executor" not in st.session_state:
    st.session_state["agent_executor"] = agent_executor

# Process user input and generate responses
if prompt := st.chat_input():
    agent_executor = st.session_state.agent_executor
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = agent_executor.invoke({"input": prompt})["output"]
    msg = {"role": "assistant", "content": response}
    st.session_state.messages.append(msg)
    st.chat_message("assistant").write(msg["content"])
    st.session_state.agent_executor = agent_executor
