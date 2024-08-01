# Ask a Github Repo
"""
To get information from a Github repository (for example, one that contains the code of a software library we want to use), 
we have to read its notes and code.
To ask a Github repo using natural language.

Load the Github repo as a collection of documents and apply the RAG technique

● Load the Github repo as a collection of text documents.
● Convert the documents into embeddings.
● Load the embeddings into a vector database.
● Create a RetrievalQA chain to retrieve the data.
"""

# Import package from parent folder
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.MyUtils import clear_terminal, logger

clear_terminal()


# Initialise foundation LLM

from utils.MyModels import init_llm, LlmModel, BaseChatModel

llm: BaseChatModel = init_llm(LlmModel.MISTRAL, temperature=0)


# Load the github repo

root_dir = "data/thefuzz-master"
document_chunks = []


# Load the text file

import os
from langchain_community.document_loaders.text import TextLoader

for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        try:
            loader = TextLoader(
                os.path.join(dirpath, file),
                encoding="utf-8"
            )
            document_chunks.extend(loader.load_and_split())
        except Exception as e:
            pass

print(f"We have {len(document_chunks)} chunks.")
print(document_chunks[0].page_content[:300])

# Convert text chunks in numeric vectors (called "embeddings")

from utils.MyEmbeddingFunction import SentenceEmbeddingFunction

my_embeddings = SentenceEmbeddingFunction()

# Load the embeddings to a vector database

from utils.MyVectorStore import chroma_from_documents

vectorstore = chroma_from_documents(
    documents=document_chunks, embedding=my_embeddings, collection_name="qa_github_repo"
)

# Create a Retrieval Question & Answering Chain

from langchain.chains import RetrievalQA

QA_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
)

# Question & Answering APP

question = """
What function do I use if I want to find 
the most similar item in a list of items?
"""

from langchain.callbacks import StdOutCallbackHandler

result = QA_chain.invoke(
    {"query": question}, config={"callbacks": [StdOutCallbackHandler()]}
)

print(result)
