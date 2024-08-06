# RAG App: QA of a Document
"""
Foundation LLMs are limited by their context window
What if we want to ask questions about a document longer than that limit?

● Load the text document with a document loader.
● Split the document into fragments with a text splitter.
● Convert the fragments into embeddings.
● Load the embeddings into a vector database.
● Create a RetrievalQA chain to retrieve the data.
"""

# Import package from parent folder
import os
import sys

sys.path.append(os.path.abspath(os.curdir))

from utils.MyUtils import clear_terminal, logger

clear_terminal()


# Initialise foundation LLM

from utils.MyModels import init_llm, LlmModel, BaseChatModel

llm: BaseChatModel = init_llm(LlmModel.MISTRAL, temperature=0)


# Load the text file

from langchain_community.document_loaders.text import TextLoader

loader = TextLoader("data/be-good-and-how-not-to-die.txt")

document = loader.load()

# The document is loaded as a Python list with metadata

print(type(document))
print(len(document))
print(document[0].metadata)
print(f"Your document has {len(document[0].page_content)} characters")

# Split the document in small chunks

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=400)

document_chunks = text_splitter.split_documents(document)

print(f"Now you have {len(document_chunks)} chunks.")

# Convert text chunks in numeric vectors (called "embeddings")

from utils.MyEmbeddingFunction import SentenceEmbeddingFunction

my_embeddings = SentenceEmbeddingFunction()

# Load the embeddings to a vector database

from utils.MyVectorStore import chroma_from_documents

vectorstore = chroma_from_documents(
    documents=document_chunks,
    embedding=my_embeddings,
    persist_directory="basic_rag_app_qa_from_docx",
    collection_name="qa_from_docx",
)

# Create a Retrieval Question & Answering Chain

from langchain.chains import RetrievalQA

QA_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
)

# Question & Answering APP

question = """
What is this article about? 
Describe it in less than 100 words.
"""

from langchain.callbacks import StdOutCallbackHandler

result = QA_chain.invoke(
    {"query": question}, config={"callbacks": [StdOutCallbackHandler()]}
)

print(result)
