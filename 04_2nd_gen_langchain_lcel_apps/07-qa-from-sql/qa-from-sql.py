# QA over SQL data
"""
Contents
● Tabular Data.
● Security Risks.
● Connect with the Database.
● Chain to convert the question into a SQL query.
● Query execution.
● QA Chain.
● Running the App.

* We will create a Q&A app over tabular data in databases.
* These app will allow us to **ask a question about the data in a database and get back a natural language answer**.
* **Building Q&A systems of SQL databases requires executing model-generated SQL queries. 
There are inherent risks in doing this**. 
Make sure that your database connection permissions are always scoped as narrowly as possible for your chain's needs.
"""


import os
import sys

sys.path.append(os.path.abspath(os.curdir))

import langchain

langchain.debug = True
langchain.verbose = True

from utils.MyUtils import logger

logger.info("running ....")


# Connect with the database

from utils.MyModels import BaseChatModel, LlmModel, init_llm

llm: BaseChatModel = init_llm(LlmModel.MISTRAL, temperature=0)


from langchain_community.utilities import SQLDatabase

sqlite_db_path = "./data/street_tree_db.sqlite"

db = SQLDatabase.from_uri(f"sqlite:///{sqlite_db_path}")


from langchain.chains import create_sql_query_chain

chain = create_sql_query_chain(llm, db)

response = chain.invoke({"question": "How many species of trees are in San Francisco?"})

print(response)

response = db.run(response)

print(response)

# * We can also inspect the chain directly for its prompts:
chain.get_prompts()[0].pretty_print()


from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

execute_query = QuerySQLDataBaseTool(db=db)

write_query = create_sql_query_chain(llm, db)

chain = write_query | execute_query

response = chain.invoke({"question": "How many species of trees are in San Francisco?"})

print(response)

"""
* Now that we've got a way to automatically generate and execute queries, we just need to **combine the original question and SQL query result to generate a final answer**.
* We can do this by passing question and result to the LLM once more:
"""

from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

chain = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | execute_query
    )
    | answer_prompt
    | llm
    | StrOutputParser()
)

response = chain.invoke({"question": "How many species of trees are in San Francisco?"})

print(response)
