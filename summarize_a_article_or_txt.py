from utils.MyUtils import logger
from utils.MyModels import init_llm, LlmModel, BaseChatModel 

########################################################################
# region Initialise foundation LLM

llm: BaseChatModel = init_llm(LlmModel.MISTRAL, temperature=0)

# endregion Initialise foundation LLM

########################################################################

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
