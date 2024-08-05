# Summarizing App: Writing Text
"""
summarize a long text from written text
Quick study of large documents

Splitter and chain to summarize

"""
# Import package from parent folder
import os
import sys

sys.path.append(os.path.abspath(os.curdir))

import streamlit as st
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAI

from utils.MyModels import BaseChatModel, LlmModel, init_llm


def generate_response(txt):
    # llm = OpenAI(
    #     temperature=0,
    #     openai_api_key=openai_api_key
    # )
    llm: BaseChatModel = init_llm(LlmModel.MISTRAL, temperature=0)
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(txt)
    docs = [Document(page_content=t) for t in texts]
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    return chain.run(docs)


st.set_page_config(page_title="Writing Text Summarization")
st.title("Writing Text Summarization")

txt_input = st.text_area("Enter your text", "", height=200)

result = []
with st.form("summarize_form", clear_on_submit=True):
    # openai_api_key = st.text_input(
    #     "OpenAI API Key", type="password", disabled=not txt_input
    # )
    submitted = st.form_submit_button("Submit")
    # if submitted and openai_api_key.startswith("sk-"):
    #     response = generate_response(txt_input)
    #     result.append(response)
    #     del openai_api_key
    response = generate_response(txt_input)
    result.append(response)

if len(result):
    st.info(response)
