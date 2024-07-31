# Ask an API
"""
To request data from an external API we have to do it by writing code.
Now we want to ask an API using natural language

Define the API and use a predefined chain to ask it questions

● Define the API: base url and endpoints.
● Create an APIChain with the API and the LLM.
● Ask the API using natural language.
"""
# https://python.langchain.com/v0.1/docs/use_cases/apis/

from utils.MyUtils import logger, clear_terminal

clear_terminal()


# Initialise foundation LLM

from utils.MyModels import init_llm, LlmModel, BaseChatModel

llm: BaseChatModel = init_llm(LlmModel.MISTRAL, temperature=0)

# Define the API documentation

api_docs = """
BASE URL: https://restcountries.com/

API Documentation:

The API endpoint /v3.1/name/{name} Used to find informatin about 
a country. All URL parameters are listed below:
    - name: Name of country - Example: Italy, France
    
The API endpoint /v3.1/currency/{currency} Used to find information 
about a region. All URL parameters are listed below:
    - currency: 3 letter currency. Example: USD, COP

The API endpoint /v3.1/lang/{language} Used to find information 
about the official language of the country. All URL parameters 
are listed below:
    - language: language of the country. Example: English, Spanish
    
"""

# Create a chain to read the API documentation

from langchain.chains import APIChain

api_chain = APIChain.from_llm_and_api_docs(
    llm=llm,
    api_docs=api_docs,
    verbose=True,
    limit_to_domains=["https://restcountries.com/"]
)

# Question & Answering APP

from langchain.callbacks import StdOutCallbackHandler

question = "Give me information about France in less than 100 words."

"""
result = api_chain.invoke(
    {"query": question}, config={"callbacks": [StdOutCallbackHandler()]}
)
"""
result = api_chain.run(question)

print(result)

question2 = """
List the top 3 biggest countries 
where the official language is French.
"""

"""
result = api_chain.invoke(
    {"query": question2}, config={"callbacks": [StdOutCallbackHandler()]}
)
"""
result = api_chain.run(question2)

print(result)

