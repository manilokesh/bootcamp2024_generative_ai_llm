# Asking From App: PDF
"""
ask about the content of a PDF file.
RAG technique
"""
import os
import sys

sys.path.append(os.path.abspath(os.curdir))

import langchain

langchain.debug = True
langchain.verbose = True

import streamlit as st

# from langchain_community.vectorstores import FAISS
from utils.MyVectorStore import chroma_from_documents, chroma_get
from utils.MyModels import LlmModel, init_llm
from langchain.document_loaders.csv_loader import CSVLoader

# from langchain_community.embeddings import OpenAIEmbeddings
from utils.MyEmbeddingFunction import SentenceEmbeddingFunction
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


# LLM and key loading function
def load_LLM(openai_api_key):
    """Logic for loading the chain you want to use should go here."""
    # Make sure your openai_api_key is set as an environment variable
    # llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    llm = init_llm(LlmModel.MISTRAL, temperature=0)
    return llm


# Page title and header
st.set_page_config(page_title="Ask from CSV File with FAQs about Napoleon")
st.header("Ask from CSV File with FAQs about Napoleon")


# Input OpenAI API Key
# def get_openai_api_key():
#     input_text = st.text_input(
#         label="OpenAI API Key ",
#         placeholder="Ex: sk-2twmA8tfCb8un4...",
#         key="openai_api_key_input",
#         type="password",
#     )
#     return input_text

# openai_api_key = get_openai_api_key()


# if openai_api_key:
# embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
embeddings = SentenceEmbeddingFunction()

vectordb_file_path = "my_vecdtordb"


def create_db():
    loader = CSVLoader(file_path="data/napoleon-faqs.csv", source_column="prompt")
    documents = loader.load()
    # vectordb = FAISS.from_documents(documents, embeddings)
    vectordb = chroma_from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="008-streamlit-ask-csv",
        # collection_name="streamlit-ask-csv",
    )

    # Save vector database locally
    # vectordb.save_local(vectordb_file_path)


def execute_chain():
    # Load the vector database from the local folder
    # vectordb = FAISS.load_local(vectordb_file_path, embedding)
    vectordb = chroma_get(
        embedding_function=embeddings, persist_directory="008-streamlit-ask-csv"
    )

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)

    template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, respond "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    llm = load_LLM(openai_api_key="openai_api_key")

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    return chain


# if __name__ == "__main__":
#     create_db()
#     chain = execute_chain()

btn = st.button("Private button: re-create database")
if btn:
    create_db()

question = st.text_input("Question: ")

if question:
    with st.spinner("Wait, please. I am working on it..."):

        chain = execute_chain()
        response = chain(question)

        st.header("Answer")
        st.write(response["result"])
