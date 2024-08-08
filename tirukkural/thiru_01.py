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
from langchain_core.output_parsers import JsonOutputParser
from thirukural_structure import ThirukuralResponse


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
        Explain first 2 (related to phrase) alone in detail, for remaining, list the kural alone with brief explanation
        Only explain if you are completely sure that the information given is accurate. 
        Refuse to explain otherwise. 
        Make a funny story to explain the topic precisely.
        Each answer/item in the list of answers should have a related emoji and kural in tamil presented in bold. 
        Format the output as a JSON object with the following JSON structure:
        {format_instructions}
        Based on the below context:\n\n{context}
    """

    # Set up the parser with the defined Pydantic model
    parser = JsonOutputParser(pydantic_object=ThirukuralResponse)

    # Create the prompt template with partial variables to include format instructions
    prompt = ChatPromptTemplate.from_messages(
        [("system", template), ("user", "{input}")]
    ).partial(format_instructions=parser.get_format_instructions())
    return prompt


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_rag_chain():
    vectorstore = get_vector_store()
    retriever = get_retriever(vectorstore)
    prompt = get_prompt_template()

    # Use Gemini or Ollama LLM
    llm: BaseChatModel = init_llm(LlmModel.GEMINI, temperature=0)

    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | JsonOutputParser()
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
    st.image("./tirukkural/data/kuralpics/1.jpg", output_format="PNG", width=400)

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

        rag_chain = get_rag_chain()
        response = rag_chain.invoke(question)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.json(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
