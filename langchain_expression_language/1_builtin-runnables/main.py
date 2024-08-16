# more advanced use of RunnableParallel

import os
import sys

sys.path.append(os.path.abspath(os.curdir))

import langchain

langchain.debug = True
langchain.verbose = True


from utils.MyModels import LlmModel, init_llm

# from langchain_community.embeddings import OpenAIEmbeddings
from utils.MyEmbeddingFunction import SentenceEmbeddingFunction

from utils.MyVectorStore import chroma_from_texts

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# vectorstore = FAISS.from_texts(
#     ["AI Accelera has trained more than 7.000 Alumni from all continents and top companies"], embedding=OpenAIEmbeddings()
# )
vectorstore = chroma_from_texts(
    texts=[
        "AI Accelera has trained more than 7.000 Alumni from all continents and top companies"
    ],
    embedding=SentenceEmbeddingFunction(),
    persist_directory="langclain_el_runnableparallel_1",
)


retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# model = ChatOpenAI(model="gpt-3.5-turbo")
model = init_llm(LlmModel.MISTRAL, temperature=0)

retrieval_chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)

retrieval_chain.invoke("who are the Alumni of AI Accelera?")
