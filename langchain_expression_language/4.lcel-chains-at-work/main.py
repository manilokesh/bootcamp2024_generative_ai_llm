# LCEL chain at work in a typical RAG app


import os
import sys

sys.path.append(os.path.abspath(os.curdir))

import langchain

langchain.debug = True
langchain.verbose = True

# --------------------------------------------------

# Connect with an LLM

from utils.MyModels import LlmModel, init_llm

model = init_llm(LlmModel.MISTRAL, temperature=0)

# --------------------------------------------------


# work with a typical RAG example

import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

splits = text_splitter.split_documents(docs)

from utils.MyVectorStore import chroma_from_documents
from utils.MyEmbeddingFunction import SentenceEmbeddingFunction

# vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
vectorstore = chroma_from_documents(
    documents=splits,
    embedding=SentenceEmbeddingFunction(),
    persist_directory="langclain_el_main_operations",
    collection_name="RunnableParallel",
)

retriever = vectorstore.as_retriever()

prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

rag_chain.invoke("What is Task Decomposition?")


# rag_chain = (
#     RunnableParallel({"context": retriever | format_docs, "question": RunnablePassthrough()})
#     | prompt
#     | model
#     | StrOutputParser()
# )

# rag_chain.invoke("What is Task Decomposition?")
