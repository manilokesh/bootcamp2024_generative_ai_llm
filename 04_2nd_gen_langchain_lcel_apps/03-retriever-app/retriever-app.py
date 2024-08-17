# to build a simple Retriever LLM App with LangChain

import os
import sys

sys.path.append(os.path.abspath(os.curdir))

import langchain

langchain.debug = True
langchain.verbose = True

from langchain_core.documents import Document

documents = [
    Document(
        page_content="John F. Kennedy served as the 35th president of the United States from 1961 until his assassination in 1963.",
        metadata={"source": "us-presidents-doc"},
    ),
    Document(
        page_content="Robert F. Kennedy was a key political figure and served as the U.S. Attorney General; he was also assassinated in 1968.",
        metadata={"source": "us-politics-doc"},
    ),
    Document(
        page_content="The Kennedy family is known for their significant influence in American politics and their extensive philanthropic efforts.",
        metadata={"source": "kennedy-family-doc"},
    ),
    Document(
        page_content="Edward M. Kennedy, often known as Ted Kennedy, was a U.S. Senator who played a major role in American legislation over several decades.",
        metadata={"source": "us-senators-doc"},
    ),
    Document(
        page_content="Jacqueline Kennedy Onassis, wife of John F. Kennedy, was an iconic First Lady known for her style, poise, and dedication to cultural and historical preservation.",
        metadata={"source": "first-lady-doc"},
    ),
]

from utils.MyEmbeddingFunction import SentenceEmbeddingFunction

my_embeddings = SentenceEmbeddingFunction()

from utils.MyVectorStore import chroma_from_documents

vectorstore = chroma_from_documents(
    documents=documents,
    embedding=my_embeddings,
    persist_directory="04_03_retriever_app",
    collection_name="qa_retrieval_chain",
)

result = vectorstore.similarity_search("John")

print(result)


result = vectorstore.similarity_search_with_score("John")

print(result)

# Retrievers

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

result = retriever.batch(["John", "Robert"])
print(result)


from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

retriever = RunnableLambda(vectorstore.similarity_search).bind(k=1)  # select top result

result = retriever.batch(["John", "Robert"])
print(result)

## Simple Retriever

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

message = """
Answer this question using the provided context only.

{question}

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([("human", message)])


from utils.MyModels import BaseChatModel, LlmModel, init_llm

llm: BaseChatModel = init_llm(LlmModel.MISTRAL, temperature=0)


chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

response = chain.invoke("tell me about Jackie")

print(response.content)
