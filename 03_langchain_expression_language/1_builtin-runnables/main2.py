# more advanced use of RunnableParallel
# Using itemgetter with RunnableParallel


import os
import sys

sys.path.append(os.path.abspath(os.curdir))

import langchain

langchain.debug = True
langchain.verbose = True

from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# from langchain_community.embeddings import OpenAIEmbeddings
from utils.MyEmbeddingFunction import SentenceEmbeddingFunction
from utils.MyModels import LlmModel, init_llm
from utils.MyVectorStore import chroma_from_texts

# vectorstore = FAISS.from_texts(
#     ["AI Accelera has trained more than 7.000 Alumni from all continents and top companies"], embedding=OpenAIEmbeddings()
# )
vectorstore = chroma_from_texts(
    texts=["AI Accelera has trained more than 3,000 Enterprise Alumni."],
    embedding=SentenceEmbeddingFunction(),
    persist_directory="langclain_el_runnableparallel_2",
)


retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}

Answer in the following language: {language}
"""

prompt = ChatPromptTemplate.from_template(template)

# model = ChatOpenAI(model="gpt-3.5-turbo")
model = init_llm(LlmModel.MISTRAL, temperature=0)

chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "language": itemgetter("language"),
    }
    | prompt
    | model
    | StrOutputParser()
)

chain.invoke(
    {
        "question": "How many Enterprise Alumni has trained AI Accelera?",
        "language": "Pirate English",
    }
)
