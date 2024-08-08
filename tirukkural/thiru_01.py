import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from utils.MyModels import BaseChatModel, LlmModel, init_llm
from utils.MyUtils import logger

## Logging ##
# clear_terminal()

from langchain_core.runnables import RunnablePassthrough
from utils.MyVectorStore import chroma_get
from utils.MyEmbeddingFunction import SentenceEmbeddingFunction
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

_DB_PRESIST_DIR = "tirukkural"
_DB_COLLECTION_NAME = "tirukkural"
embeddings = SentenceEmbeddingFunction()


def get_vector_store():
    vectorstore = chroma_get(
        embedding_function=embeddings,
        persist_directory=_DB_PRESIST_DIR,
        collection_name=_DB_COLLECTION_NAME,
    )

    return vectorstore


def get_retriever(vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="similarity",  # Also test "similarity", "mmr"
        search_kwargs={"k": 8},
    )
    return retriever


def get_prompt_template():

    template = """
        Generate responses based on the provided context. 
        The phrase might be given in english or tamil
        The phrase might be a Thirukural or a phrase related to Thirukural 
        To extract the thirukkural from the context, Get matching thirukkural related to phrase provided in the context
        If you are not able to determine anything based on the phrase, tell its not a thirukural
        Explain first 2 (related to phrse) alone in detail, for remaining, list the kural alone with brief explanation
        Only explain if you are completely sure that the information given is accurate. 
        Refuse to explain otherwise. 
        Make a funny story to explain the topic precisly 
        Each answer / item in list of answers, should have a related emoji and kural in tamil presented in bold 
        Format the output as bullet-points text with the following keys:
        - Kural: [kural number] with Kural in tamil
        - Kural: [kural number] with Kural in English
        - Actual explantion :
            - English
            - Tamil [multiple authors name and explanation]
        -  பால் : [english translation]
            - English
            - Tamil
        - அதிகாரம்/Chapter/Athigaram: [english translation]
            - Chapter No 
            - English
            - Tamil
        - Story : 
            - English
            - Tamil
        - Other Matching Kural(s) :
            - English
            - Tamil
        based on the below context:\n\n{context}
    """

    prompt = ChatPromptTemplate.from_messages(
        [("system", template), ("user", "{input}")]
    )
    return prompt


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_rag_chain():
    vectorstore = get_vector_store()
    retriever = get_retriever(vectorstore)
    prompt = get_prompt_template()

    # use gemini or ollama
    llm: BaseChatModel = init_llm(LlmModel.GEMINI, temperature=0)

    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def main():
    rag_chain = get_rag_chain()
    response = rag_chain.invoke("கடவுள் வாழ்த்து")
    logger.info(response)


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


from streamlit_navigation_bar import st_navbar

page = st_navbar(
    ["Questions on Kural", "Thirukural Book", "About"], options={"use_padding": False}
)
# st.write(page)

question = st.chat_input(
    placeholder="Kural in English or tamil",
    key=None,
    max_chars=None,
    disabled=False,
    on_submit=None,
    args=None,
    kwargs=None,
)


# # Streamed response emulator
# def response_generator():
#     response = random.choice(
#         [
#             "Hello there 🙏 How can I assist you today?",
#             "Hi 🙏, Is there anything I can help you with?",
#             "🙏 Do you need help?",
#         ]
#     )
#     for word in response.split():
#         yield word + " "
#         time.sleep(0.05)


# # Display assistant response in chat message container
# with st.chat_message("assistant"):
#     response = st.write_stream(response_generator())

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
        rag_chain = get_rag_chain()
        response = rag_chain.invoke(question)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})


# """
# #### Note: what does the previos formatter function do?
# The `format_docs` function takes a list of objects named `docs`. Each object in this list is expected to have an attribute named `page_content`, which stores textual content for each document.

# The purpose of the function is to extract the `page_content` from each document in the `docs` list and then combine these contents into a single string. The contents of different documents are separated by two newline characters (`\n\n`), which means there will be an empty line between the content of each document in the final string. This formatting choice makes the combined content easier to read by clearly separating the content of different documents.

# Here's a breakdown of how the function works:
# 1. The `for doc in docs` part iterates over each object in the `docs` list.
# 2. For each iteration, `doc.page_content` accesses the `page_content` attribute of the current document, which contains its textual content.
# 3. The `join` method then takes these pieces of text and concatenates them into a single string, inserting `\n\n` between each piece to ensure they are separated by a blank line in the final result.

# The function ultimately returns this newly formatted single string containing all the document contents, neatly separated by blank lines.
# """
