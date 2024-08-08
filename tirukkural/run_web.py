import os
import sys

sys.path.append(os.path.abspath(os.curdir))

import langchain

langchain.debug = True
langchain.verbose = True

_DB_PRESIST_DIR = "tirukkural"
_DB_COLLECTION_NAME = "tirukkural"

from utils.MyVectorStore import chroma_get

# from langchain.chains.retrieval_qa.base import RetrievalQA
from utils.MyModels import LlmModel, init_llm
from utils.MyEmbeddingFunction import SentenceEmbeddingFunction
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from utils.MyModels import BaseChatModel, LlmModel, init_llm


def get_vectordb():

    embeddings = SentenceEmbeddingFunction()

    chromadb = chroma_get(
        embedding_function=embeddings,
        persist_directory=_DB_PRESIST_DIR,
        collection_name=_DB_COLLECTION_NAME,
    )
    return chromadb


# Set up retriever chain
def setup_retriever_chain(llm, retriever):
    system_template = """
        Given the above conversation, generate a search query to look up to get information relevant to the conversation        
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("user", "{input}"),
            ("user", system_template),
        ]
    )
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain


# Set up document chain
def setup_document_chain(llm):
    template = """
        You are a expert tamil pandit and english phd who can give a immense explanation and story for thirukural in both english and tamil
        based on the below context. 
        The thirukural might be given in english or tamil
        If the given is not a thirukural or any explanation related to thirukural, tell its not a thirukural
        If the user phrase matches to any thirukural, then organise the list of kurals matching
        If multiple items matches to the phrase, explain first 2 alone in detail, for remaining, list the kural alone following the explanation why only kural is presented
        Only explain if you are completely sure that the information given is accurate. 
        Refuse to explain otherwise. 
        Make sure your explanation are detailed , if possible with image clips. 
        Include from which which "அதிகாரம்/Chapter:"
        Make a story explain the topic precisly 
        Each answer / item in list of answers, should have a related emoji and kural in tamil presented in bold 
        Format the output as bullet-points text with the following keys:
        - Kural: [kural number] with Kural in tamil
        - Actual explantion :
            - English
            - Tamil [multiple authors name and explanation]
        -  பால் : [english translation]
            - English
            - Tamil
        - அதிகாரம்/Chapter/Athigaram: [english translation]
            - English
            - Tamil
        - Story : 
            - English
            - Tamil
        based on the below context:\n\n{context}
    """

    prompt = ChatPromptTemplate.from_messages(
        [("system", template), ("user", "{input}")]
    )
    document_chain = create_stuff_documents_chain(llm, prompt)
    return document_chain


# Set up QA chain
def setup_qa_chain(retriever_chain, document_chain):
    qa = create_retrieval_chain(retriever_chain, document_chain)
    return qa


def create_qa(vectdb):

    retriever = vectdb.as_retriever(
        search_type="mmr",  # Also test "similarity"
        search_kwargs={"k": 8},
    )

    # use gemini or ollama
    llm: BaseChatModel = init_llm(LlmModel.MISTRAL, temperature=0)

    retriever_chain = setup_retriever_chain(llm, retriever)
    document_chain = setup_document_chain(llm)
    qa = setup_qa_chain(retriever_chain, document_chain)
    return qa


vectdb = get_vectordb()
qa = create_qa(vectdb)

# def main():
#     question = "ஒழுக்கம் விழுப்பந் தரலான் ஒழுக்கம் உயிரினும் ஓம்பப் படும்"
#     result = qa.invoke({"input": question})
#     print(result.get("answer"))
#     # for chunk in qa.stream({"input": question}):
#     #     if answer_chunk := chunk.get("answer"):
#     #         print(answer_chunk)
#
# if __name__ == "__main__":
#     main()


import streamlit as st

st.set_page_config(
    page_title="Thirukural",
    page_icon="🙏",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)
st.header("Thirukural")


question = st.chat_input(
    placeholder="Kural in English or tamil",
    key=None,
    max_chars=None,
    disabled=False,
    on_submit=None,
    args=None,
    kwargs=None,
)

import random
import time


# Streamed response emulator
def response_generator():
    response = random.choice(
        [
            "Hello there 🙏 How can I assist you today?",
            "Hi 🙏, Is there anything I can help you with?",
            "🙏 Do you need help?",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


# Display assistant response in chat message container
with st.chat_message("assistant"):
    response = st.write_stream(response_generator())

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps

if question and len(question) > 0:

    # Display user message in chat message container
    st.chat_message("user").markdown(question)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("Wait, please. Looking for manuscripts 📜 ..."):
        # result = qa.invoke({"input": question})

        # # st.write_stream(qa.stream({"input": kural_input}))
        # with st.chat_message("user"):
        #     st.write(question)
        # result = qa.invoke({"input": question})
        # with st.chat_message("அ"):
        #     st.markdown(result.get("answer"))
        # for chunk in qa.stream({"input": question}):
        #     if answer_chunk := chunk.get("answer"):
        #         st.write(answer_chunk)

        response = qa.invoke({"input": question})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response.get("answer"))

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
