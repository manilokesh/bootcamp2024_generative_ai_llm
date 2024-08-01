# Ask a Database
"""
instead of complex languages like SQL,  want to ask a database using natural language.
create an SQLDatabaseChain with the database and the LLM to which we can ask questions

● Load the database.
● Create the SQLDatabaseChain.
● Ask questions in natural language.
"""

# Import package from parent folder
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.MyUtils import clear_terminal, logger

clear_terminal()

from utils.MyModels import BaseChatModel, LlmModel, init_llm

llm: BaseChatModel = init_llm(LlmModel.MISTRAL, temperature=0)


# Load SQLite database
 
from langchain_community.utilities.sql_database import SQLDatabase 

sqlite_db_path = "data/street_tree_db.sqlite"

db = SQLDatabase.from_uri(f"sqlite:///{sqlite_db_path}")

# Create a chain with de LLM and the database
   
"""
from langchain_community.utilities import SQLDatabaseChain
db_chain = SQLDatabaseChain.from_llm(
    llm=llm,
    database=db,
    verbose=True
)
"""
from langchain_experimental.sql import SQLDatabaseChain

db_chain = SQLDatabaseChain(
    llm=llm,
    database=db,
    verbose=True
)

db_chain.invoke("How many species of trees are in San Francisco?")


db_chain.invoke("How many trees of the species Ficus nitida are there in San Francisco?")