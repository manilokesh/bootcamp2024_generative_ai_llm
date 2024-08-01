#Summarize a Long Article
"""
Using a predefined LangChain chain to help the Foundation LLM, summarize a document that exceeds the context window limit. 

1. Load the document.
2. Check its token count.
3. Split it into smaller parts.
4. Use a predefined LangChain chain to send the parts to ChatGPT and get a summary of the document.
"""

# Import package from parent folder
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.MyUtils import clear_terminal, logger

clear_terminal()


from utils.MyModels import init_llm, LlmModel, BaseChatModel 

llm: BaseChatModel = init_llm(LlmModel.MISTRAL, temperature=0)
 
# Load the text file

with open("data/be-good-and-how-not-to-die.txt", "r") as file:
    article = file.read()

print(type(article))

# Print the first 285 characters of the article

print(article[:285])

# Check how many tokens are in the article

num_tokens = llm.get_num_tokens(article)
print(f"There are {num_tokens} in the article.")

# Split the article in smaller chunks

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n"], chunk_size=5000, chunk_overlap=350
)

article_chunks = text_splitter.create_documents([article])

print(f"You have {len(article_chunks)} chunks instead of 1 article.")


# Use a chain to help the LLM to summarize the 8 chunks

from langchain.chains.summarize import load_summarize_chain
from langchain.callbacks import StdOutCallbackHandler

chain = load_summarize_chain(llm=llm, chain_type="map_reduce")

article_summary = chain.invoke(
    {"input_documents": article_chunks}, config={"callbacks": [StdOutCallbackHandler()]}
)

print(article_summary)
