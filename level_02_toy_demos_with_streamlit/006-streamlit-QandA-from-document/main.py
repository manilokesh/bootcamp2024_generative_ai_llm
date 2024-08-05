# Asking From App: PDF
"""
ask about the content of a PDF file.

Research, quick study of large documents, etc.
RAG technique
"""

# Import package from parent folder
import os
import sys

sys.path.append(os.path.abspath(os.curdir))

# __import__("pysqlite3")
import sys

import chromadb

# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter

# import sqlite3
from langchain_openai import OpenAI
from PyPDF2 import PdfReader

# from langchain.embeddings.openai import OpenAIEmbeddings
from utils.MyEmbeddingFunction import SentenceEmbeddingFunction
from utils.MyModels import BaseChatModel, LlmModel, init_llm

# from langchain_community.vectorstores import FAISS
from utils.MyVectorStore import chroma_from_documents

# Input .txt file
# Format file
# Split file
# Create embeddings
# Store embeddings in vector store
# Input query
# Run QA chain
# Output


def generate_response(file, openai_api_key, query):
    # format file
    reader = PdfReader(file)
    formatted_document = []
    for page in reader.pages:
        formatted_document.append(page.extract_text())
    # split file
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.create_documents(formatted_document)
    # create embeddings
    # embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    embeddings = SentenceEmbeddingFunction()
    # load to vector database
    # store = Chroma.from_documents(texts, embeddings)

    # store = FAISS.from_documents(docs, docs)
    store = chroma_from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="streamlit-qa-from-document",
        collection_name="streamlit-QandA-from-document",
    )

    # create retrieval chain
    retrieval_chain = RetrievalQA.from_chain_type(
        # llm=OpenAI(temperature=0, openai_api_key=openai_api_key),
        llm=init_llm(LlmModel.MISTRAL, temperature=0),
        chain_type="stuff",
        retriever=store.as_retriever(),
    )
    # run chain with query
    return retrieval_chain.run(query)


st.set_page_config(page_title="Q&A from a long PDF Document")
st.title("Q&A from a long PDF Document")

uploaded_file = st.file_uploader("Upload a .pdf document", type="pdf")

query_text = st.text_input(
    "Enter your question:",
    placeholder="Write your question here",
    disabled=not uploaded_file,
)

result = []
with st.form("myform", clear_on_submit=True):
    # openai_api_key = st.text_input(
    #     "OpenAI API Key:", type="password", disabled=not (uploaded_file and query_text)
    # )
    submitted = st.form_submit_button(
        "Submit", disabled=not (uploaded_file and query_text)
    )
    # if submitted and openai_api_key.startswith("sk-"):
    #     with st.spinner("Wait, please. I am working on it..."):
    #         response = generate_response(uploaded_file, openai_api_key, query_text)
    #         result.append(response)
    #         del openai_api_key
    if submitted:
        with st.spinner("Wait, please. I am working on it..."):
            response = generate_response(uploaded_file, "openai_api_key", query_text)
            result.append(response)

if len(result):
    st.info(response)
