# https://www.datacamp.com/tutorial/run-llama-3-locally


# Import package from parent folder
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.MyUtils import clear_terminal, logger

# pip install python-magic-bin

from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader("data", glob="HIMS_Functional.docx")
books = loader.load()
print(len(books))


from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(books)


"""
vectorstore = Chroma.from_documents(
    documents=all_splits,
    embedding=OllamaEmbeddings(model=LlmModel.MISTRAL.value, show_progress=True),
    persist_directory="./chroma_db",
)
"""

from utils.MyEmbeddingFunction import SentenceEmbeddingFunction

my_embeddings = SentenceEmbeddingFunction()

from utils.MyVectorStore import chroma_from_documents

vectorstore = chroma_from_documents(
    documents=all_splits, embedding=my_embeddings, collection_name="qa_retrieval_chain"
)

question = "how are patients registered?"
docs = vectorstore.similarity_search(question)
print(docs)
