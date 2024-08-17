# How to build a simple Agent LLM App with LangGraph


import os
import sys

sys.path.append(os.path.abspath(os.curdir))

import langchain

langchain.debug = True
langchain.verbose = True

from utils.MyUtils import logger


logger.info("running ....")


## Define tools

from langchain_community.tools.tavily_search import TavilySearchResults

search = TavilySearchResults(max_results=2)
result = search.invoke("Who are the top stars of the Olympics 2024?")

print(result)

tools = [search]

# In order to enable this model to do tool calling we use .bind_tools

from utils.MyModels import BaseChatModel, LlmModel, init_llm

llm: BaseChatModel = init_llm(LlmModel.MISTRAL, temperature=0)

llm_with_tools = llm.bind_tools(tools)

# Create the agent
"""
* **Note that below we are passing in the model, not llm_with_tools**. 
That is because create_tool_calling_executor will call .bind_tools for us under the hood.
"""

from langgraph.prebuilt import chat_agent_executor

agent_executor = chat_agent_executor.create_tool_calling_executor(llm, tools)

# Run the agent
from langchain_core.messages import HumanMessage

response = agent_executor.invoke(
    {"messages": [HumanMessage(content="Where is the Hockey Olympics 2024 played?")]}
)

print(response["messages"])

# for chunk in agent_executor.stream(
#     {"messages": [HumanMessage(content="When and where will it be the 2024 Eurocup final match?")]}
# ):
#     print(chunk)
#     print("----")
