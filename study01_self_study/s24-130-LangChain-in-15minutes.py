
# Import package from parent folder
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

########################################################################
# region Initialise foundation LLM

from utils.MyModels import init_llm, LlmModel, BaseChatModel

llm: BaseChatModel = init_llm(LlmModel.GEMINI)

# endregion Initialise foundation LLM

########################################################################

# result = llm.invoke("What are the 5 best things to do in life?")
# print(result)

########################################################################

from langchain_core.prompts import PromptTemplate

my_prompt_template = PromptTemplate(
    input_variables=["destination"],
    template="What are the 3 best things to do in {destination}?",
)

user_input = "Barcelona"

# print(llm.invoke(my_prompt_template.format(destination=user_input)))

########################################################################

# > pip install playwright
# > playwright install

# https://towardsdatascience.com/building-a-math-application-with-langchain-agents-23919d09a4d3

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.tools import BaseTool
from langchain.chains import LLMMathChain
from playwright.sync_api import sync_playwright


class SearchWebTool(BaseTool):
    name = "search_from_web"
    description = "Searchs on the web for the specified query"

    def _run(self, query: str):
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(f"https://www.google.com/search?q={query}")
            results = page.query_selector_all("h3")
            titles = [result.inner_text() for result in results]
            browser.close()
            return f"Top results for '{query}':\n" + "\n".join(titles[:5])

    def _arun(self, ticker: str):
        raise NotImplementedError("This tool does not support async")


problem_chain = LLMMathChain.from_llm(llm=llm)
math_tool = Tool.from_function(
    name="Calculator",
    func=problem_chain.run,
    description="""Useful for when you need to answer questions about math. This tool is only for math questions and nothing else. Only input math expressions.""",
)

tools = [SearchWebTool(), math_tool]

prompt = hub.pull("hwchase17/react")

# Create the agent executor
agent = create_react_agent(tools=tools, llm=llm, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)

# Use the agent
question = "Who is the current leader of India? What is the largest prime number that is smaller than their age?"
response = agent_executor.invoke({"input": question})
print(response)

########################################################################
