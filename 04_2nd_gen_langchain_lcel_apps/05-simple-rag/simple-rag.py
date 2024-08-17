# How to build a simple RAG LLM App with LangChain
"""
● Install ChromaDB.
● Load private document.
● Splitter.
● Vector Store with embeddings.
● Retriever.
● Prompt.
● Document Formatter.
● RAG Chain.
● Run App.
"""

import os
import sys

sys.path.append(os.path.abspath(os.curdir))

import langchain

langchain.debug = True
langchain.verbose = True

from utils.MyUtils import logger


logger.info("running ....")


from langchain_community.document_loaders import TextLoader

loader = TextLoader("./data/be-good.txt")
docs = loader.load()


from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)


from utils.MyEmbeddingFunction import SentenceEmbeddingFunction
from utils.MyVectorStore import chroma_from_documents

vectorstore = chroma_from_documents(
    documents=splits,
    embedding=SentenceEmbeddingFunction(),
    persist_directory="04_05-simple-rag",
    collection_name="qa_retrieval_chain",
)
retriever = vectorstore.as_retriever()


from langchain import hub

prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


from utils.MyModels import BaseChatModel, LlmModel, init_llm

llm: BaseChatModel = init_llm(LlmModel.MISTRAL, temperature=0)


from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
result = rag_chain.invoke("What is this article about?")
print(result)
