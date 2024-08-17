# QA over PDF file
"""
● Load PDF file.
● RAG.
● Citations.
"""

import os
import sys

sys.path.append(os.path.abspath(os.curdir))

import langchain

langchain.debug = True
langchain.verbose = True

from utils.MyUtils import logger

logger.info("running ....")

from langchain_community.document_loaders import PyPDFLoader

file_path = "./data/Be_Good.pdf"

loader = PyPDFLoader(file_path)

docs = loader.load()

print(len(docs))
print(docs[0].page_content[0:100])
print(docs[0].metadata)


from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

splits = text_splitter.split_documents(docs)


from utils.MyEmbeddingFunction import SentenceEmbeddingFunction
from utils.MyVectorStore import chroma_from_documents

vectorstore = chroma_from_documents(
    documents=splits,
    embedding=SentenceEmbeddingFunction(),
    persist_directory="04_08-qa-from-pdf",
    collection_name="qa_retrieval_chain",
)

retriever = vectorstore.as_retriever()


from langchain_core.prompts import ChatPromptTemplate

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


from utils.MyModels import BaseChatModel, LlmModel, init_llm

llm: BaseChatModel = init_llm(LlmModel.MISTRAL, temperature=0)

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

question_answer_chain = create_stuff_documents_chain(llm, prompt)

rag_chain = create_retrieval_chain(retriever, question_answer_chain)

results = rag_chain.invoke({"input": "What is this article about?"})

print(results["answer"])
